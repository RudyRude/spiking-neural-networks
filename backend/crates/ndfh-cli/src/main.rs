use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use ndfh_api::{HeCreate, InMemoryTxn, TxnApi};
use ndfh_hdx::io as hdx_io;
use ndfh_hdx::DatasetManifest;
use ndfh_hgts::AsOfEngine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "ndfh", version, about = "NDF-H CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Convert data from legacy formats to NDF-H (placeholder)
    Convert {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
    },

    /// Validate a dataset.yaml manifest (basic checks)
    Verify {
        #[arg(short, long)]
        manifest: String,
        /// Optional path to JSON Schema file for strict validation
        #[arg(long)]
        schema: Option<String>,
        /// Additionally verify shard checksums (blake3) relative to the manifest directory
        #[arg(long, default_value_t = false)]
        check_checksums: bool,
        /// Optional path to a security policy YAML to validate against the policy schema
        #[arg(long)]
        policy: Option<String>,
    },

    /// Run evaluation/benchmarks on a dataset (placeholder)
    Eval {
        #[arg(short, long)]
        dataset: String,
    },

    /// Inspect a manifest and print a summary
    Inspect {
        #[arg(short, long)]
        manifest: String,
    },

    /// Demonstrate AS OF snapshot using in-memory membership/catalog
    AsOfDemo {
        /// Event time to snapshot at (nanoseconds)
        #[arg(long, default_value_t = 150_i64)]
        t_ns: i64,
    },

    /// Export snapshot(s) to compatibility formats
    Export(ExportCmd),
}

#[derive(Debug, Clone, ValueEnum)]
enum ExportFormat {
    LpgGraphml,
    LpgJson,
    RdfNquads,
}

#[derive(Args, Debug)]
struct ExportCmd {
    /// Path to dataset root (directory containing dataset.yaml), or a path to dataset.yaml
    #[arg(short = 'd', long)]
    dataset: String,
    /// Snapshot time (nanoseconds)
    #[arg(long, default_value_t = 150_i64)]
    as_of: i64,
    /// Output directory
    #[arg(short, long, default_value = "./out")]
    out: String,
    /// Export format
    #[arg(short = 'f', long, value_enum, default_value_t = ExportFormat::LpgGraphml)]
    format: ExportFormat,
    /// Include labels as properties when available
    #[arg(long, default_value_t = false)]
    include_labels: bool,
    /// Filter by head vertex id
    #[arg(long)]
    filter_head: Option<u64>,
    /// Optional path to a security policy file to enforce before export
    #[arg(long)]
    policy: Option<String>,
    /// Purpose of use (for ABAC)
    #[arg(long)]
    purpose: Option<String>,
}

fn main() -> Result<()> {
    // Initialize tracing/logging (stdout by default; OTEL stdout when feature is enabled in ndfh-api)
    __ndfh_cli_init_tracing();

    let cli = Cli::parse();
    match cli.command {
        Commands::Convert { input, output } => {
            // Build manifest from directory and write dataset.yaml
            let input_path = std::path::Path::new(&input);
            let out_path = std::path::Path::new(&output);
            let out_file = if out_path.extension().is_some() {
                out_path.to_path_buf()
            } else {
                out_path.join("dataset.yaml")
            };
            let mf = ndfh_hdx::DatasetManifest::build_from_dir(
                input_path,
                "converted-ndfh",
                "0.1.0",
                "NDF-H 1.0",
            )
            .with_context(|| format!("failed to build manifest from {}", input))?;
            mf.write_to_path(&out_file)
                .with_context(|| format!("failed to write manifest to {}", out_file.display()))?;
            println!("Wrote manifest to {}", out_file.display());
        }
        Commands::Verify {
            manifest,
            schema,
            check_checksums,
            policy,
        } => {
            let mf = DatasetManifest::from_path(&manifest)
                .with_context(|| format!("failed to read manifest: {}", manifest))?;
            mf.validate_basic()
                .with_context(|| "basic validation failed".to_string())?;

            // If --schema provided, use it; otherwise, attempt default path if present.
            if let Some(schema_path) = schema {
                match DatasetManifest::validate_against_schema(&manifest, &schema_path) {
                    Ok(_) => println!("Schema validation OK with {}", schema_path),
                    Err(e) => println!("Schema validation skipped or failed: {}", e),
                }
            } else {
                let default_schema = "schemas/ndfh/hdx/dataset.manifest.schema.json";
                if std::path::Path::new(default_schema).exists() {
                    match DatasetManifest::validate_against_schema(&manifest, default_schema) {
                        Ok(_) => println!("Schema validation OK with {}", default_schema),
                        Err(e) => println!("Schema validation skipped or failed: {}", e),
                    }
                } else {
                    println!("Schema file not found at default path; skipped schema validation");
                }
            }

            // Optional checksum verification
            if check_checksums {
                let root = std::path::Path::new(&manifest)
                    .parent()
                    .unwrap_or(std::path::Path::new("."));
                match mf.verify_checksums(root) {
                    Ok(mismatches) => {
                        if mismatches.is_empty() {
                            println!("Checksum verification OK ({} shards)", mf.shards.len());
                        } else {
                            println!("Checksum mismatches ({}):", mismatches.len());
                            for m in mismatches {
                                println!("  - {}", m);
                            }
                        }
                    }
                    Err(e) => {
                        println!("Checksum verification error: {}", e);
                    }
                }
            }

            // Optional security policy validation against schema if provided or present next to manifest
            {
                let policy_path = if let Some(p) = policy {
                    Some(std::path::PathBuf::from(p))
                } else {
                    let default = std::path::Path::new(&manifest)
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join("security.policy.yaml");
                    if default.exists() {
                        Some(default)
                    } else {
                        None
                    }
                };

                if let Some(ppath) = policy_path {
                    let policy_schema_default = "schemas/ndfh/hdx/security.policy.schema.json";
                    if std::path::Path::new(policy_schema_default).exists() {
                        match ndfh_hdx::DatasetManifest::validate_against_schema(
                            &ppath,
                            policy_schema_default,
                        ) {
                            Ok(_) => println!(
                                "Security policy validation OK with {}",
                                policy_schema_default
                            ),
                            Err(e) => println!("Security policy validation failed: {}", e),
                        }
                    } else {
                        println!(
                            "Policy schema not found at {}; skipped policy validation",
                            policy_schema_default
                        );
                    }
                }
            }

            // Print detected conformance level (heuristic)
            let level = mf.detect_conformance();
            println!("Manifest OK: {} (conformance {:?})", mf.dataset_name, level);
        }
        Commands::Eval { dataset } => {
            // Load and inspect dataset for basic evaluation metrics
            let ds_path = PathBuf::from(&dataset);
            let manifest_path: PathBuf = if ds_path.is_dir() {
                ds_path.join("dataset.yaml")
            } else {
                ds_path.clone()
            };
            if !manifest_path.exists() {
                bail!("dataset manifest not found at {}", manifest_path.display());
            }
            let mf = DatasetManifest::from_path(&manifest_path)
                .with_context(|| format!("failed to read manifest: {}", manifest_path.display()))?;
            mf.validate_basic()
                .context("manifest basic validation failed")?;

            // Basic evaluation: conformance level and dataset statistics
            let conformance = mf.detect_conformance();
            println!("Dataset: {} v{}", mf.dataset_name, mf.dataset_version);
            println!("Conformance Level: {:?}", conformance);
            println!("NDF Version: {}", mf.ndf_version);
            println!("License: {}", mf.license);
            println!("Shards: {}", mf.shards.len());

            // Compute time range and row counts if available
            let mut total_rows = 0u64;
            let mut min_time: Option<i64> = None;
            let mut max_time: Option<i64> = None;

            for shard in mf.shards.values() {
                // Consolidated fields (post-schema normalization)
                total_rows += shard.num_rows;
                let (start, end) = shard.time_range;
                min_time = Some(min_time.map(|t| t.min(start)).unwrap_or(start));
                max_time = Some(max_time.map(|t| t.max(end)).unwrap_or(end));
            }

            println!("Total Rows: {}", total_rows);
            if let (Some(min), Some(max)) = (min_time, max_time) {
                println!("Time Range: {} to {} ({} ns span)", min, max, max - min);
            }

            // PII classification summary
            if let Some(pii_max) = pii_max_class(&mf) {
                println!("Max PII Classification: {}", pii_max);
            }
            if let Some(pp) = &mf.pii_policy {
                if let Some(class) = &pp.classification {
                    println!("Dataset PII Policy: {}", class);
                }
            }
        }
        Commands::Inspect { manifest } => {
            let mf = DatasetManifest::from_path(&manifest)
                .with_context(|| format!("failed to read manifest: {}", manifest))?;
            println!("{}", mf.summary());
            println!("Detected conformance: {:?}", mf.detect_conformance());
        }
        Commands::AsOfDemo { t_ns } => {
            // Build a simple in-memory topology: one hyperedge with head 99 and tails 10,11 active at t_ns=150
            let mut txn = InMemoryTxn::default();
            let h_id = txn.he_create(HeCreate {
                head_v: 99,
                fe_spec_json: "{}".to_string(),
                state_schema_json: None,
            })?;
            // add tails and a removed tail
            txn.mem_add(h_id, 10, 100)?;
            txn.mem_add(h_id, 11, 120)?;
            txn.mem_add(h_id, 12, 90)?;
            txn.mem_rem(h_id, 12, 110)?;
            let net = txn.snapshot_as_of(t_ns);
            let ids = net.hyperedge_ids();
            println!("AS OF t_ns={} -> hyperedges: {}", t_ns, ids.len());
            if let Some(first) = ids.first() {
                if let Some(edge) = net.get_hyperedge(*first) {
                    println!(
                        "edge {}: sources={}, targets={}",
                        first.raw(),
                        edge.sources.len(),
                        edge.targets.len()
                    );
                }
            }
        }
        Commands::Export(cmd) => {
            // Resolve dataset.yaml path
            let ds_path = PathBuf::from(&cmd.dataset);
            let manifest_path: PathBuf = if ds_path.is_dir() {
                ds_path.join("dataset.yaml")
            } else {
                ds_path.clone()
            };
            if !manifest_path.exists() {
                bail!("dataset manifest not found at {}", manifest_path.display());
            }
            let mf = DatasetManifest::from_path(&manifest_path)
                .with_context(|| format!("failed to read manifest: {}", manifest_path.display()))?;
            mf.validate_basic()
                .context("manifest basic validation failed")?;

            // Subject roles: currently assumed ["exporter"] until CLI accepts --role flags.
            let subject_roles = vec!["exporter".to_string()];

            // If a policy file is provided OR found next to the manifest, evaluate it; otherwise fall back to minimal built-in enforcement.
            let default_policy_path = manifest_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("security.policy.yaml");
            let effective_policy_path = cmd.policy.as_ref().map(PathBuf::from).or_else(|| {
                if default_policy_path.exists() {
                    Some(default_policy_path.clone())
                } else {
                    None
                }
            });

            if let Some(ppath) = effective_policy_path {
                let policy = load_security_policy(&ppath).with_context(|| {
                    format!("failed to read security policy: {}", ppath.display())
                })?;
                // Resource table reflects the export format to enable per-format rules
                let resource_table = match cmd.format {
                    ExportFormat::LpgGraphml => "lpg-graphml",
                    ExportFormat::LpgJson => "lpg-json",
                    ExportFormat::RdfNquads => "rdf-nquads",
                };
                let decision = evaluate_policy(
                    &policy,
                    &mf,
                    &subject_roles,
                    cmd.purpose.as_deref(),
                    "export",
                    resource_table,
                );
                match decision {
                    Decision::Deny(reason) => {
                        ndfh_api::observability::record_policy_decision("deny", Some(&reason));
                        bail!("export denied by policy: {}", reason);
                    }
                    Decision::Allow => {
                        ndfh_api::observability::record_policy_decision("allow", None);
                    }
                }
            } else {
                enforce_export_policy(&mf, &subject_roles, cmd.purpose.as_deref())
                    .context("export blocked by minimal policy")?;
            }

            // Observability: start timer for export latency
            let t_start = Instant::now();

            // Build a snapshot from storage if possible, otherwise fall back to deterministic demo snapshot
            let dataset_root = manifest_path.parent().unwrap_or(std::path::Path::new("."));
            let mem_log_opt = hdx_io::load_membership_log_from_manifest(&mf, dataset_root)
                .with_context(|| "failed to load membership shards")?;
            let cat_opt = hdx_io::load_hyperedge_catalog_from_manifest(&mf, dataset_root)
                .with_context(|| "failed to load hyperedges shards")?;

            let net = if let (Some(mem_log), Some(cat)) = (mem_log_opt, cat_opt) {
                // Real AS OF snapshot from manifest-backed shards
                let snapshot = AsOfEngine::snapshot_with_catalog(&mem_log, &cat, cmd.as_of);
                ndfh_api::observability::record_snapshot_metrics(
                    cmd.as_of,
                    snapshot.hyperedge_ids().len(),
                    "storage",
                );
                snapshot
            } else {
                // Deterministic demo snapshot (when shards are absent)
                let mut txn = InMemoryTxn::default();
                let seed_head = 99u64;
                let h_id = txn.he_create(HeCreate {
                    head_v: seed_head,
                    fe_spec_json: "{}".to_string(),
                    state_schema_json: None,
                })?;
                txn.mem_add(h_id, 10, cmd.as_of - 50)?;
                txn.mem_add(h_id, 11, cmd.as_of - 30)?;
                txn.mem_add(h_id, 12, cmd.as_of - 60)?;
                txn.mem_rem(h_id, 12, cmd.as_of - 40)?;
                let snapshot = txn.snapshot_as_of(cmd.as_of);
                ndfh_api::observability::record_snapshot_metrics(
                    cmd.as_of,
                    snapshot.hyperedge_ids().len(),
                    "demo",
                );
                snapshot
            };
            // Record total hyperedges before any filtering (for metrics)
            let orig_total_hyperedges = net.hyperedge_ids().len();

            // license_permits_derivatives moved to top-level helper below to avoid duplication

            /// Security policy structures (minimal evaluator)
            #[derive(Debug, Clone, Serialize, Deserialize)]
            struct SecurityPolicy {
                #[serde(default)]
                rules: Vec<PolicyRule>,
            }

            #[derive(Debug, Clone, Serialize, Deserialize)]
            struct PolicyRule {
                id: Option<String>,
                description: Option<String>,
                #[serde(default)]
                r#match: BTreeMap<String, serde_yaml::Value>,
                effect: String, // "allow" | "deny"
            }

            enum Decision {
                Allow,
                Deny(String),
            }

            fn load_security_policy(path: &Path) -> Result<SecurityPolicy> {
                let s = std::fs::read_to_string(path)?;
                let p: SecurityPolicy = serde_yaml::from_str(&s)?;
                Ok(p)
            }

            /// Evaluate minimal policy by exact/contains matching on a small vocabulary:
            /// - subject.roles: [..]
            /// - action: "export"
            /// - context.purpose: string
            /// - resource.license.permits_derivatives: bool
            /// - resource.pii_max_class: "none"|"low"|"moderate"|"high"
            /// - resource.pii_class: per-export sensitivity (here equal to pii_max_class)
            /// - resource.table: export target ("lpg-graphml"|"lpg-json"|"rdf-nquads")
            fn evaluate_policy(
                policy: &SecurityPolicy,
                mf: &DatasetManifest,
                subject_roles: &[String],
                purpose: Option<&str>,
                action: &str,
                resource_table: &str,
            ) -> Decision {
                // Build context
                let resource_license = mf.license.clone();
                let derivatives = license_permits_derivatives(&resource_license);
                let pii_max = pii_max_class(mf).unwrap_or_else(|| "none".to_string());
                // Expose a single-class view aligned to the most sensitive shard class
                let resource_pii_class = pii_max.clone();

                // Helper to test a rule's match constraints
                let mut matched_allow = None::<String>;
                for rule in &policy.rules {
                    let mut ok = true;
                    for (k, v) in &rule.r#match {
                        match k.as_str() {
                            "action" => {
                                // Support string or sequence
                                if let Some(exp) = v.as_str() {
                                    ok = ok && (exp == action);
                                } else if let Some(arr) = v.as_sequence() {
                                    let mut any = false;
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            if s == action {
                                                any = true;
                                                break;
                                            }
                                        }
                                    }
                                    ok = ok && any;
                                } else {
                                    ok = false;
                                }
                            }
                            "context.purpose" => {
                                // Support string or sequence
                                if let Some(exp) = v.as_str() {
                                    ok = ok && (Some(exp) == purpose);
                                } else if let Some(arr) = v.as_sequence() {
                                    let mut any = false;
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            if Some(s) == purpose {
                                                any = true;
                                                break;
                                            }
                                        }
                                    }
                                    ok = ok && any;
                                } else {
                                    ok = false;
                                }
                            }
                            "subject.roles" => {
                                // Expect sequence of strings; require all present in subject_roles (subset)
                                if let Some(arr) = v.as_sequence() {
                                    for item in arr {
                                        if let Some(role) = item.as_str() {
                                            if !subject_roles.iter().any(|r| r == role) {
                                                ok = false;
                                                break;
                                            }
                                        } else {
                                            ok = false;
                                            break;
                                        }
                                    }
                                } else {
                                    ok = false;
                                }
                            }
                            "resource.license.permits_derivatives" => {
                                if let Some(b) = v.as_bool() {
                                    ok = ok && (derivatives == b);
                                } else {
                                    ok = false;
                                }
                            }
                            "resource.pii_max_class" => {
                                if let Some(exp) = v.as_str() {
                                    ok = ok && (pii_max == exp);
                                } else if let Some(arr) = v.as_sequence() {
                                    let mut any = false;
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            if pii_max == s {
                                                any = true;
                                                break;
                                            }
                                        }
                                    }
                                    ok = ok && any;
                                } else {
                                    ok = false;
                                }
                            }
                            "resource.pii_class" => {
                                if let Some(exp) = v.as_str() {
                                    ok = ok && (resource_pii_class == exp);
                                } else if let Some(arr) = v.as_sequence() {
                                    let mut any = false;
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            if resource_pii_class == s {
                                                any = true;
                                                break;
                                            }
                                        }
                                    }
                                    ok = ok && any;
                                } else {
                                    ok = false;
                                }
                            }
                            "resource.table" => {
                                if let Some(exp) = v.as_str() {
                                    ok = ok && (resource_table == exp);
                                } else if let Some(arr) = v.as_sequence() {
                                    let mut any = false;
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            if resource_table == s {
                                                any = true;
                                                break;
                                            }
                                        }
                                    }
                                    ok = ok && any;
                                } else {
                                    ok = false;
                                }
                            }
                            _ => {
                                // Unknown field: treat as non-match
                                ok = false;
                            }
                        }
                        if !ok {
                            break;
                        }
                    }

                    if ok {
                        let rid = rule.id.clone().unwrap_or_else(|| "<unnamed>".into());
                        if rule.effect.eq_ignore_ascii_case("deny") {
                            return Decision::Deny(rid);
                        } else if rule.effect.eq_ignore_ascii_case("allow") {
                            matched_allow = Some(rid);
                            // keep scanning in case a later deny should take precedence
                        }
                    }
                }

                // Default decision: if a policy is provided but no allow matched, deny by default.
                match matched_allow {
                    Some(_) => Decision::Allow,
                    None => Decision::Deny("no-allowing-rule-matched".into()),
                }
            }

            // Ensure output directory
            fs::create_dir_all(&cmd.out).with_context(|| format!("creating {}", cmd.out))?;

            // Determine hyperedges to export with optional head filter
            let mut allowed_ids: Vec<u32> =
                net.hyperedge_ids().into_iter().map(|h| h.raw()).collect();
            allowed_ids.sort_unstable();
            if let Some(head_filter) = cmd.filter_head {
                allowed_ids.retain(|hid| {
                    if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(*hid)) {
                        edge.targets.iter().any(|t| t.raw() as u64 == head_filter)
                    } else {
                        false
                    }
                });
            }

            // Attempt to propagate LICENSE/COPYING file from dataset root into export bundle
            let dataset_root = manifest_path.parent().unwrap_or(std::path::Path::new("."));
            let license_candidates = [
                "LICENSE",
                "LICENSE.txt",
                "LICENSE.md",
                "COPYING",
                "COPYING.txt",
                "COPYING.md",
            ];
            let mut copied_license: Option<String> = None;
            for cand in &license_candidates {
                let src = dataset_root.join(cand);
                if src.exists() {
                    let dst = Path::new(&cmd.out).join("LICENSE.txt");
                    // Best-effort copy; do not fail export if copy fails
                    if fs::copy(&src, &dst).is_ok() {
                        copied_license = Some(cand.to_string());
                        break;
                    }
                }
            }

            // Emit NOTICE (metadata with latency is written after export below)
            let notice = format!(
                "NDF-H Export NOTICE\n\
                 Dataset: {name} v{ver} (NDF {ndf})\n\
                 License: {lic}\n\
                 Purpose: {purpose}\n\
                 AS OF: {asof}\n\
                 Generated: {ts}\n\
                 LicenseFileCopied: {copied}\n",
                name = mf.dataset_name,
                ver = mf.dataset_version,
                ndf = mf.ndf_version,
                lic = mf.license,
                purpose = cmd.purpose.clone().unwrap_or_else(|| "unspecified".into()),
                asof = cmd.as_of,
                ts = chrono::Utc::now().to_rfc3339(),
                copied = copied_license.unwrap_or_else(|| "none".into()),
            );
            let notice_path = Path::new(&cmd.out).join("NOTICE.txt");
            fs::write(&notice_path, notice)
                .with_context(|| format!("writing {}", notice_path.display()))?;

            // Deterministic export
            match cmd.format {
                ExportFormat::LpgGraphml => {
                    let s = encode_graphml(&net, cmd.include_labels, Some(&allowed_ids));
                    let out = Path::new(&cmd.out).join("snapshot.graphml");
                    fs::write(&out, s).with_context(|| format!("writing {}", out.display()))?;
                    println!("GraphML export -> {}", out.display());
                }
                ExportFormat::LpgJson => {
                    let s = encode_lpg_json(&net, cmd.include_labels, Some(&allowed_ids));
                    let out = Path::new(&cmd.out).join("snapshot.lpg.json");
                    fs::write(&out, s).with_context(|| format!("writing {}", out.display()))?;
                    println!("LPG JSON export -> {}", out.display());
                }
                ExportFormat::RdfNquads => {
                    let s = encode_rdf_nquads(&net, Some(&allowed_ids));
                    let out = Path::new(&cmd.out).join("snapshot.nq");
                    fs::write(&out, s).with_context(|| format!("writing {}", out.display()))?;
                    println!("RDF N-Quads export -> {}", out.display());
                }
            }

            // Observability: finalize metrics and write export.meta.json
            let exported_hyperedges = allowed_ids.len();
            let filtered_count = if orig_total_hyperedges >= exported_hyperedges {
                (orig_total_hyperedges - exported_hyperedges) as u64
            } else {
                0
            };
            let latency_ms: u64 = t_start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);

            let export_meta = serde_json::json!({
                "dataset_name": mf.dataset_name,
                "dataset_version": mf.dataset_version,
                "ndf_version": mf.ndf_version,
                "license": mf.license,
                "purpose": cmd.purpose,
                "as_of": cmd.as_of,
                "format": match cmd.format { ExportFormat::LpgGraphml => "lpg-graphml", ExportFormat::LpgJson => "lpg-json", ExportFormat::RdfNquads => "rdf-nquads" },
                "filter_head": cmd.filter_head,
                "metrics": {
                    "hyperedges_total": orig_total_hyperedges as u64,
                    "hyperedges_exported": exported_hyperedges as u64,
                    "filtered_count": filtered_count,
                    "latency_ms": latency_ms
                }
            });
            let meta_path = Path::new(&cmd.out).join("export.meta.json");
            fs::write(
                &meta_path,
                serde_json::to_string_pretty(&export_meta).unwrap(),
            )
            .with_context(|| format!("writing {}", meta_path.display()))?;

            // Also emit metrics via observability hook (tracing; OTLP-ready)
            ndfh_api::observability::record_export_metrics(
                orig_total_hyperedges as u64,
                exported_hyperedges as u64,
                filtered_count,
                latency_ms,
            );
        }
    }
    ndfh_api::observability::shutdown_tracer();
    Ok(())
}

/// Initialize tracing/logging once at process start using ndfh-api helper.
/// This is done at the earliest entry to main to allow downstream crates to emit spans if enabled.
#[doc(hidden)]
fn __ndfh_cli_init_tracing() {
    // Safe to call multiple times; tracing-subscriber handles global set only once.
    // We avoid depending on tracing macros in this crate by just initializing.
    ndfh_api::observability::init_tracer();
}

/// Minimal ABAC enforcement for exporters:
/// - Require subject role "exporter"
/// - If dataset-level PII classification is "moderate" or "high" and purpose == "demo", deny.
/// - Also deny demo if any shard pii_class is "moderate" or "high".
fn enforce_export_policy(
    mf: &DatasetManifest,
    subject_roles: &[String],
    purpose: Option<&str>,
) -> Result<()> {
    // role check
    let has_exporter = subject_roles.iter().any(|r| r == "exporter");
    if !has_exporter {
        bail!("subject lacks required role 'exporter'");
    }
    // dataset pii policy check
    if let Some(pp) = &mf.pii_policy {
        if let Some(class) = pp.classification.as_deref() {
            let sensitive = matches!(class, "moderate" | "high");
            if sensitive && matches!(purpose, Some("demo")) {
                bail!(
                    "dataset-level PII classification={} incompatible with purpose=demo",
                    class
                );
            }
        }
    }
    // shard pii_class check (deny demo if any shard is moderate/high)
    if matches!(purpose, Some("demo")) {
        if let Some(max_class) = pii_max_class(mf) {
            if matches!(max_class.as_str(), "moderate" | "high") {
                bail!(
                    "shard-level PII classification={} incompatible with purpose=demo",
                    max_class
                );
            }
        }
    }
    // Additional conservative gating without a policy: if max shard class is "high",
    // only allow strictly internal/audit purposes.
    if let Some(max_class) = pii_max_class(mf) {
        if max_class == "high" {
            let p = purpose.unwrap_or("unspecified").to_lowercase();
            let allowed_internal = p == "internal" || p == "audit";
            if !allowed_internal {
                bail!(
                    "high PII classification requires purpose=internal|audit (got '{}')",
                    p
                );
            }
        }
    }

    // Minimal license gating when no external policy is provided:
    // - If license does NOT permit derivatives, deny export for outward-facing purposes.
    //   Allow only if explicitly marked internal/audit.
    let spdx = mf.license.trim().to_uppercase();
    let permits_derivatives = if spdx.starts_with("CC-BY-ND") {
        false
    } else {
        matches!(
            spdx.as_str(),
            "CC-BY-4.0" | "CC0-1.0" | "MIT" | "APACHE-2.0" | "BSD-3-CLAUSE" | "BSD-2-CLAUSE"
        )
    };
    if !permits_derivatives {
        let p = purpose.unwrap_or("unspecified").to_lowercase();
        let allowed_internal = p == "internal" || p == "audit";
        if !allowed_internal {
            bail!(
                "license {} does not permit derivatives; export requires purpose=internal|audit",
                mf.license
            );
        }
    }

    ndfh_api::observability::shutdown_tracer();
    Ok(())
}

/// Compute the maximum pii_class across shards (none < low < moderate < high)
fn pii_max_class(mf: &DatasetManifest) -> Option<String> {
    fn score(s: &str) -> i32 {
        match s {
            "none" => 0,
            "low" => 1,
            "moderate" => 2,
            "high" => 3,
            _ => -1,
        }
    }
    let mut max_s: Option<(&str, i32)> = None;
    for shard in mf.shards.values() {
        if let Some(class) = shard.pii_class.as_deref() {
            let sc = score(class);
            if sc >= 0 && max_s.map(|(_, x)| sc > x).unwrap_or(true) {
                max_s = Some((class, sc));
            }
        }
    }
    max_s.map(|(c, _)| c.to_string())
}

/// Simple license mapping to derived property "permits_derivatives"
fn license_permits_derivatives(spdx: &str) -> bool {
    // Conservative mapping for common cases
    // - CC-BY-4.0, CC0-1.0, MIT, Apache-2.0 permit derivatives
    // - CC-BY-ND* do not permit derivatives
    // - Unknown licenses default to false
    let s = spdx.trim().to_uppercase();
    if s.starts_with("CC-BY-ND") {
        return false;
    }
    matches!(
        s.as_str(),
        "CC-BY-4.0" | "CC0-1.0" | "MIT" | "APACHE-2.0" | "BSD-3-CLAUSE" | "BSD-2-CLAUSE"
    )
}

/// Deterministic GraphML encoder (ManyToOne edges; hyperedge node reification)
fn encode_graphml(
    net: &ndfh_core::HypergraphNetwork,
    _include_labels: bool,
    allowed_hids: Option<&[u32]>,
) -> String {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    let mut buf = String::new();
    buf.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    buf.push('\n');
    buf.push_str(r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#);
    buf.push('\n');
    buf.push_str(r#"<graph edgedefault="directed">"#);
    buf.push('\n');

    // Determine hyperedges to emit
    let mut hedge_ids: Vec<u32> = match allowed_hids {
        Some(slice) => slice.to_vec(),
        None => net.hyperedge_ids().into_iter().map(|h| h.raw()).collect(),
    };
    hedge_ids.sort_unstable();

    // Collect vertex ids from edges
    let mut vertex_ids: BTreeSet<u32> = BTreeSet::new();
    for h in &hedge_ids {
        if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(*h)) {
            for s in &edge.sources {
                vertex_ids.insert(s.raw());
            }
            for t in &edge.targets {
                vertex_ids.insert(t.raw());
            }
        }
    }

    // Emit vertex nodes
    for v in vertex_ids {
        let _ = write!(buf, r#"<node id="v{}"/>"#, v);
        buf.push('\n');
    }
    // Reify each hyperedge as node "h{ID}", connect sources->h and h->target
    for h in hedge_ids {
        let _ = write!(buf, r#"<node id="h{}"/>"#, h);
        buf.push('\n');
        if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(h)) {
            for s in &edge.sources {
                let _ = write!(buf, r#"<edge source="v{}" target="h{}"/>"#, s.raw(), h);
                buf.push('\n');
            }
            for t in &edge.targets {
                let _ = write!(buf, r#"<edge source="h{}" target="v{}"/>"#, h, t.raw());
                buf.push('\n');
            }
        }
    }

    buf.push_str("</graph>\n</graphml>\n");
    buf
}

/// Deterministic LPG JSON encoder: { "nodes": [ {id: "vX"}...], "edges": [ {src, dst, kind}... ] }
fn encode_lpg_json(
    net: &ndfh_core::HypergraphNetwork,
    _include_labels: bool,
    allowed_hids: Option<&[u32]>,
) -> String {
    use serde_json::json;
    let mut node_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut edges: Vec<serde_json::Value> = Vec::new();

    let mut hedge_ids: Vec<u32> = match allowed_hids {
        Some(slice) => slice.to_vec(),
        None => net.hyperedge_ids().into_iter().map(|h| h.raw()).collect(),
    };
    hedge_ids.sort_unstable();

    for h in hedge_ids {
        node_set.insert(format!("h{}", h));
        if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(h)) {
            for s in &edge.sources {
                node_set.insert(format!("v{}", s.raw()));
                edges.push(json!({"src": format!("v{}", s.raw()), "dst": format!("h{}", h), "kind": "tail"}));
            }
            for t in &edge.targets {
                node_set.insert(format!("v{}", t.raw()));
                edges.push(json!({"src": format!("h{}", h), "dst": format!("v{}", t.raw()), "kind": "head"}));
            }
        }
    }

    let nodes: Vec<serde_json::Value> = node_set.into_iter().map(|id| json!({"id": id})).collect();
    serde_json::to_string_pretty(&json!({"nodes": nodes, "edges": edges}))
        .unwrap_or_else(|_| "{}".to_string())
}

/// Deterministic RDF N-Quads encoder using simple vocabulary:
/// <hedge:h{H}> <ndfh:hasTail> <vertex:v{V}> .
/// <hedge:h{H}> <ndfh:hasHead> <vertex:v{V}> .
fn encode_rdf_nquads(net: &ndfh_core::HypergraphNetwork, allowed_hids: Option<&[u32]>) -> String {
    let base = "https://ndfh.example.org/vocab/";
    let mut lines: Vec<String> = Vec::new();

    let mut hedge_ids: Vec<u32> = match allowed_hids {
        Some(slice) => slice.to_vec(),
        None => net.hyperedge_ids().into_iter().map(|h| h.raw()).collect(),
    };
    hedge_ids.sort_unstable();

    for h in hedge_ids {
        if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(h)) {
            for s in &edge.sources {
                lines.push(format!(
                    "<{}hedge/h{}> <{}hasTail> <{}vertex/v{}> .",
                    base,
                    h,
                    base,
                    base,
                    s.raw()
                ));
            }
            for t in &edge.targets {
                lines.push(format!(
                    "<{}hedge/h{}> <{}hasHead> <{}vertex/v{}> .",
                    base,
                    h,
                    base,
                    base,
                    t.raw()
                ));
            }
        }
    }

    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndfh_api::{HeCreate, InMemoryTxn};

    fn build_demo_snapshot(as_of: i64) -> ndfh_core::HypergraphNetwork {
        let mut txn = InMemoryTxn::default();
        let h_id = txn
            .he_create(HeCreate {
                head_v: 99,
                fe_spec_json: "{}".to_string(),
                state_schema_json: None,
            })
            .expect("he_create");
        txn.mem_add(h_id, 10, as_of - 50).expect("mem_add 10");
        txn.mem_add(h_id, 11, as_of - 30).expect("mem_add 11");
        txn.mem_add(h_id, 12, as_of - 60).expect("mem_add 12");
        txn.mem_rem(h_id, 12, as_of - 40).expect("mem_rem 12");
        txn.snapshot_as_of(as_of)
    }

    #[test]
    fn graphml_encoder_is_deterministic() {
        let net = build_demo_snapshot(150);
        let allowed_ids: Vec<u32> = {
            let mut ids: Vec<u32> = net.hyperedge_ids().into_iter().map(|h| h.raw()).collect();
            ids.sort_unstable();
            ids
        };
        let s1 = encode_graphml(&net, false, Some(&allowed_ids));
        let s2 = encode_graphml(&net, false, Some(&allowed_ids));
        assert_eq!(
            s1, s2,
            "GraphML encoder output must be byte-stable for same snapshot"
        );
    }

    #[test]
    fn lpg_json_encoder_is_deterministic() {
        let net = build_demo_snapshot(150);
        let mut allowed_ids: Vec<u32> = net.hyperedge_ids().into_iter().map(|h| h.raw()).collect();
        allowed_ids.sort_unstable();
        let s1 = encode_lpg_json(&net, false, Some(&allowed_ids));
        let s2 = encode_lpg_json(&net, false, Some(&allowed_ids));
        assert_eq!(
            s1, s2,
            "LPG JSON encoder output must be byte-stable for same snapshot"
        );
    }

    #[test]
    fn rdf_nquads_encoder_is_deterministic() {
        let net = build_demo_snapshot(150);
        let mut allowed_ids: Vec<u32> = net.hyperedge_ids().into_iter().map(|h| h.raw()).collect();
        allowed_ids.sort_unstable();
        let s1 = encode_rdf_nquads(&net, Some(&allowed_ids));
        let s2 = encode_rdf_nquads(&net, Some(&allowed_ids));
        assert_eq!(
            s1, s2,
            "RDF N-Quads encoder output must be byte-stable for same snapshot"
        );
    }

    #[test]
    fn head_filter_effect_is_consistent() {
        let net = build_demo_snapshot(150);
        // Build allowed_ids filtered by head 99
        let mut filtered: Vec<u32> = net.hyperedge_ids().into_iter().map(|h| h.raw()).collect();
        filtered.sort_unstable();
        filtered.retain(|hid| {
            if let Some(edge) = net.get_hyperedge(ndfh_core::HyperedgeId::from(*hid)) {
                edge.targets.iter().any(|t| t.raw() as u64 == 99)
            } else {
                false
            }
        });
        let s = encode_lpg_json(&net, false, Some(&filtered));
        // Ensure that when we pass the already filtered list again, we get the same bytes (idempotent filtering)
        let s_again = encode_lpg_json(&net, false, Some(&filtered));
        assert_eq!(
            s, s_again,
            "Filtering by head and re-encoding should be stable and idempotent"
        );
    }
}
