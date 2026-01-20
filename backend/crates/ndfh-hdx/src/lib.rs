#![allow(dead_code)]
//! NDF-H HDX: dataset manifest and packaging skeleton.

pub mod io;

use serde::{Deserialize, Serialize};
#[cfg(feature = "schema-validate")]
use serde_json::Value as JsonValue;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

/// Dataset-level PII declaration aligned with HDX manifest schema.
/// Fields are optional except `classification` when present in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PiiPolicy {
    pub classification: Option<String>, // "none" | "low" | "moderate" | "high"
    #[serde(default)]
    pub basis: Option<String>,
    #[serde(default)]
    pub reviewer: Option<String>,
    #[serde(default)]
    pub reviewed_at: Option<String>, // RFC3339
    #[serde(default)]
    pub notes: Option<String>,
}
/// Path diff helper (minimal dependency via pathdiff)
mod pathdiff {
    use std::path::{Component, Path, PathBuf};

    /// Compute relative path from base to path. Simple fallback if std::path::Path::strip_prefix fails across volumes.
    pub fn diff_paths(path: &Path, base: &Path) -> Option<PathBuf> {
        if let Ok(rel) = path.strip_prefix(base) {
            return Some(rel.to_path_buf());
        }
        // Fallback: build relative by skipping common prefix
        let mut ita = path.components();
        let mut itb = base.components();

        let mut comps_a: Vec<Component> = ita.by_ref().collect();
        let mut comps_b: Vec<Component> = itb.by_ref().collect();

        while !comps_a.is_empty() && !comps_b.is_empty() && comps_a[0] == comps_b[0] {
            comps_a.remove(0);
            comps_b.remove(0);
        }

        let mut rel = PathBuf::new();
        for _ in comps_b {
            rel.push("..");
        }
        for c in comps_a {
            rel.push(c.as_os_str());
        }
        Some(rel)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceLevel {
    L0, // events + labels only
    L1, // L0 + fire + hyperedges
    Unknown,
}

#[derive(Debug, thiserror::Error)]
pub enum HdxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Manifest validation error: {0}")]
    Validation(String),
}

pub type HdxResult<T> = Result<T, HdxError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMeta {
    pub path: String,
    pub table: String,
    pub checksum: String,
    pub time_range: (i64, i64),
    pub num_rows: u64,
    #[serde(default)]
    pub pii_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetManifest {
    pub dataset_name: String,
    pub dataset_version: String,
    pub ndf_version: String,
    pub schema_versions: BTreeMap<String, String>,
    pub license: String,
    #[serde(default)]
    pub pii_policy: Option<PiiPolicy>,
    #[serde(default)]
    pub splits: BTreeMap<String, Vec<String>>,
    pub shards: BTreeMap<String, ShardMeta>,
}

impl DatasetManifest {
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Load and parse a dataset.yaml from a path
    pub fn from_path<P: AsRef<Path>>(path: P) -> HdxResult<Self> {
        let data = fs::read_to_string(path)?;
        let mf: DatasetManifest = serde_yaml::from_str(&data)?;
        Ok(mf)
    }

    /// Perform minimal structural validation
    pub fn validate_basic(&self) -> HdxResult<()> {
        if self.dataset_name.trim().is_empty() {
            return Err(HdxError::Validation(
                "dataset_name must not be empty".into(),
            ));
        }
        if self.ndf_version.trim().is_empty() {
            return Err(HdxError::Validation("ndf_version must not be empty".into()));
        }
        if self.shards.is_empty() {
            return Err(HdxError::Validation("shards must not be empty".into()));
        }
        // Check time_range ordering
        for (sid, shard) in &self.shards {
            if shard.time_range.0 > shard.time_range.1 {
                return Err(HdxError::Validation(format!(
                    "shard {} has inverted time_range",
                    sid
                )));
            }
        }
        Ok(())
    }

    /// Validate the manifest YAML against a JSON Schema file (2020-12 compatible)
    #[cfg(feature = "schema-validate")]
    pub fn validate_against_schema<P1: AsRef<Path>, P2: AsRef<Path>>(
        manifest_path: P1,
        schema_path: P2,
    ) -> HdxResult<()> {
        let manifest_str = fs::read_to_string(&manifest_path)?;
        let schema_str = fs::read_to_string(&schema_path)?;

        // Parse schema and manifest to serde_json::Value
        let schema_json: JsonValue = serde_json::from_str(&schema_str)
            .map_err(|e| HdxError::Validation(format!("Invalid JSON schema: {e}")))?;
        // YAML to JSON value
        let manifest_yaml: serde_yaml::Value = serde_yaml::from_str(&manifest_str)
            .map_err(|e| HdxError::Validation(format!("Invalid YAML manifest: {e}")))?;
        let manifest_json: JsonValue = serde_json::to_value(manifest_yaml)
            .map_err(|e| HdxError::Validation(format!("YAML->JSON conversion failed: {e}")))?;

        // Compile and validate (use crate default draft for compatibility with 0.17)
        let compiled = jsonschema::JSONSchema::options()
            .compile(&schema_json)
            .map_err(|e| HdxError::Validation(format!("Schema compilation failed: {e}")))?;

        let result = compiled.validate(&manifest_json);
        if let Err(errors) = result {
            let mut msgs = Vec::new();
            for err in errors {
                msgs.push(format!("at {}: {}", err.instance_path, err));
            }
            return Err(HdxError::Validation(format!(
                "Schema validation errors:\n{}",
                msgs.join("\n")
            )));
        }
        Ok(())
    }

    #[cfg(not(feature = "schema-validate"))]
    pub fn validate_against_schema<P1: AsRef<Path>, P2: AsRef<Path>>(
        _manifest_path: P1,
        _schema_path: P2,
    ) -> HdxResult<()> {
        Err(HdxError::Validation(
            "schema validation feature not enabled".into(),
        ))
    }

    /// Compute a summary string for human inspection
    pub fn summary(&self) -> String {
        let mut by_table: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
        for shard in self.shards.values() {
            let entry = by_table.entry(shard.table.as_str()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += shard.num_rows;
        }
        let mut parts = Vec::new();
        for (table, (count, rows)) in by_table {
            parts.push(format!("{}: {} shards, {} rows", table, count, rows));
        }
        format!(
            "{} v{} (NDF {}) — {} shards\n{}",
            self.dataset_name,
            self.dataset_version,
            self.ndf_version,
            self.shards.len(),
            parts.join("\n")
        )
    }

    /// Verify shard checksums relative to a dataset root directory.
    /// Returns a list of mismatched shard IDs (empty means all OK).
    pub fn verify_checksums(&self, root: &Path) -> HdxResult<Vec<String>> {
        let mut mismatches = Vec::new();
        for (shard_id, meta) in &self.shards {
            let file_path = root.join(&meta.path);
            // Only support blake3:... prefix for now
            let expected = meta.checksum.trim();
            let (algo, exp_hex) = expected.split_once(':').unwrap_or(("unknown", expected));
            if algo != "blake3" {
                // Skip unsupported checksum algorithms gracefully
                continue;
            }
            // Read file and compute blake3
            let Ok(bytes) = fs::read(&file_path) else {
                mismatches.push(format!(
                    "{} (missing file {})",
                    shard_id,
                    file_path.display()
                ));
                continue;
            };
            let got = blake3::hash(&bytes).to_hex().to_string();
            if got != exp_hex {
                mismatches.push(format!(
                    "{} (expected blake3:{}, got blake3:{})",
                    shard_id, exp_hex, got
                ));
            }
        }
        Ok(mismatches)
    }

    /// Build a DatasetManifest by scanning an input directory for JSONL shards.
    /// Heuristics:
    /// - Recognizes tables by filename containing "events", "fire", or "labels"
    /// - Computes time_range from t_ns fields per line and row counts
    /// - Computes blake3 checksum of each file content
    /// - Uses relative paths (relative to input_dir)
    pub fn build_from_dir(
        input_dir: &Path,
        dataset_name: &str,
        dataset_version: &str,
        ndf_version: &str,
    ) -> HdxResult<Self> {
        let mut mf = DatasetManifest {
            dataset_name: dataset_name.to_string(),
            dataset_version: dataset_version.to_string(),
            ndf_version: ndf_version.to_string(),
            schema_versions: BTreeMap::new(),
            license: "UNSPECIFIED".to_string(),
            pii_policy: None,
            splits: BTreeMap::new(),
            shards: BTreeMap::new(),
        };

        for entry in WalkDir::new(input_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            // Consider only .jsonl inputs for this lightweight builder
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext != "jsonl" {
                    continue;
                }
            } else {
                continue;
            }

            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_lowercase();
            let table = if file_name.contains("events") {
                "events"
            } else if file_name.contains("fire") {
                "fire"
            } else if file_name.contains("labels") {
                "labels"
            } else {
                // Skip unrecognized jsonl
                continue;
            };

            let (tmin, tmax, rows) = Self::compute_time_range_and_rows(path)?;
            let checksum = Self::blake3_file(path)?;
            // Relative path from input_dir
            let rel_path = pathdiff::diff_paths(path, input_dir)
                .unwrap_or_else(|| PathBuf::from(file_name.clone()));
            let rel_str = rel_path.to_string_lossy().to_string();

            // Derive shard id from relative path
            let shard_id = rel_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&file_name)
                .to_string();

            mf.shards.insert(
                shard_id,
                ShardMeta {
                    path: rel_str,
                    table: table.to_string(),
                    checksum: format!("blake3:{}", checksum),
                    time_range: (tmin, tmax),
                    num_rows: rows,
                    pii_class: None,
                },
            );
        }

        if mf.shards.is_empty() {
            return Err(HdxError::Validation("no recognizable shards found (expected *events*.jsonl, *fire*.jsonl, or *labels*.jsonl)".into()));
        }

        Ok(mf)
    }

    /// Write manifest YAML to a file path; creates parents as needed.
    pub fn write_to_path(&self, out_path: &Path) -> HdxResult<()> {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let s = self.to_yaml().map_err(HdxError::from)?;
        fs::write(out_path, s)?;
        Ok(())
    }

    fn compute_time_range_and_rows(path: &Path) -> HdxResult<(i64, i64, u64)> {
        use std::io::{BufRead, BufReader};
        let f = fs::File::open(path)?;
        let reader = BufReader::new(f);

        let mut tmin: Option<i64> = None;
        let mut tmax: Option<i64> = None;
        let mut rows: u64 = 0;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue, // skip malformed lines
            };
            let t_ns_opt = v
                .get("t_ns")
                .and_then(|t| t.as_i64().or_else(|| t.as_f64().map(|f| f as i64)));
            if let Some(t) = t_ns_opt {
                tmin = Some(tmin.map(|x| x.min(t)).unwrap_or(t));
                tmax = Some(tmax.map(|x| x.max(t)).unwrap_or(t));
            }
            rows += 1;
        }

        match (tmin, tmax) {
            (Some(a), Some(b)) => Ok((a, b, rows)),
            _ => Err(HdxError::Validation(format!(
                "could not compute time_range for {}",
                path.display()
            ))),
        }
    }

    fn blake3_file(path: &Path) -> HdxResult<String> {
        let data = fs::read(path)?;
        Ok(blake3::hash(&data).to_hex().to_string())
    }

    /// Heuristic conformance detection
    pub fn detect_conformance(&self) -> ConformanceLevel {
        let mut has_events = false;
        let mut has_labels = false;
        let mut has_fire = false;
        let mut has_hyperedges = false;

        for shard in self.shards.values() {
            match shard.table.as_str() {
                "events" => has_events = true,
                "labels" => has_labels = true,
                "fire" => has_fire = true,
                "hyperedges" => has_hyperedges = true,
                _ => {}
            }
        }

        if has_events && has_labels {
            if has_fire && has_hyperedges {
                return ConformanceLevel::L1;
            }
            return ConformanceLevel::L0;
        }

        ConformanceLevel::Unknown
    }
}
