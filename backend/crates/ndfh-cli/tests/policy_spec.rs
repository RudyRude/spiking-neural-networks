use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use tempfile::tempdir;

// Resolve repository root from this crate's manifest directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures_dir() -> PathBuf {
    repo_root().join("examples/ndfh/fixtures")
}

fn dataset_path() -> PathBuf {
    fixtures_dir().join("dataset.sample.yaml")
}

fn policy_path() -> PathBuf {
    fixtures_dir().join("security.policy.yaml")
}

fn snapshot_name_for_format(fmt: &str) -> &'static str {
    match fmt {
        "lpg-graphml" => "snapshot.graphml",
        "lpg-json" => "snapshot.lpg.json",
        "rdf-nquads" => "snapshot.nq",
        _ => panic!("unsupported format: {}", fmt),
    }
}

#[test]
fn export_with_policy_allows_research_lpg_json() {
    let out = tempdir().expect("tempdir");
    let out_path = out.path().to_path_buf();

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("ndfh-cli");
    cmd.args([
        "export",
        "--dataset",
        &dataset_path().to_string_lossy(),
        "--as-of",
        "150",
        "--format",
        "lpg-json",
        "--out",
        &out_path.to_string_lossy(),
        "--policy",
        &policy_path().to_string_lossy(),
        "--purpose",
        "research",
    ]);

    // Expect success.
    cmd.assert().success();

    // Artifacts: snapshot + NOTICE.txt + export.meta.json
    let snapshot = out_path.join(snapshot_name_for_format("lpg-json"));
    assert!(
        snapshot.exists(),
        "expected snapshot at {}",
        snapshot.display()
    );

    let notice = out_path.join("NOTICE.txt");
    assert!(
        notice.exists(),
        "expected NOTICE.txt at {}",
        notice.display()
    );

    let meta = out_path.join("export.meta.json");
    assert!(
        meta.exists(),
        "expected export.meta.json at {}",
        meta.display()
    );
}

#[test]
fn export_with_policy_denies_demo() {
    let out = tempdir().expect("tempdir");
    let out_path = out.path().to_path_buf();

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("ndfh-cli");
    cmd.args([
        "export",
        "--dataset",
        &dataset_path().to_string_lossy(),
        "--as-of",
        "150",
        "--format",
        "lpg-graphml",
        "--out",
        &out_path.to_string_lossy(),
        "--policy",
        &policy_path().to_string_lossy(),
        "--purpose",
        "demo",
    ]);

    // Expect policy denial (non-zero).
    cmd.assert().failure();

    // Denial occurs before artifact writes; ensure no meta or snapshot was produced.
    let meta = out_path.join("export.meta.json");
    assert!(
        !meta.exists(),
        "unexpected export.meta.json present at {}",
        meta.display()
    );
    let snapshot = out_path.join(snapshot_name_for_format("lpg-graphml"));
    assert!(
        !snapshot.exists(),
        "unexpected snapshot present at {}",
        snapshot.display()
    );
}

#[test]
fn export_no_policy_allows_internal_lpg_json() {
    let out = tempdir().expect("tempdir");
    let out_path = out.path().to_path_buf();

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("ndfh-cli");
    cmd.args([
        "export",
        "--dataset",
        &dataset_path().to_string_lossy(),
        "--as-of",
        "150",
        "--format",
        "lpg-json",
        "--out",
        &out_path.to_string_lossy(),
        "--purpose",
        "internal",
    ]);

    // Expect success (built-in gates allow internal under all license cases).
    cmd.assert().success();

    // Artifacts should be present.
    let snapshot = out_path.join(snapshot_name_for_format("lpg-json"));
    assert!(
        snapshot.exists(),
        "expected snapshot at {}",
        snapshot.display()
    );
    let notice = out_path.join("NOTICE.txt");
    assert!(
        notice.exists(),
        "expected NOTICE.txt at {}",
        notice.display()
    );
    let meta = out_path.join("export.meta.json");
    assert!(
        meta.exists(),
        "expected export.meta.json at {}",
        meta.display()
    );
}

#[test]
fn export_meta_format_matches_format_arg_graphml() {
    let out = tempdir().expect("tempdir");
    let out_path = out.path().to_path_buf();

    let format = "lpg-graphml";

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("ndfh-cli");
    cmd.args([
        "export",
        "--dataset",
        &dataset_path().to_string_lossy(),
        "--as-of",
        "150",
        "--format",
        format,
        "--out",
        &out_path.to_string_lossy(),
        "--purpose",
        "internal",
    ]);

    cmd.assert().success();

    // Parse export.meta.json and assert the format field equals the expected policy resource.table mapping.
    let meta_path = out_path.join("export.meta.json");
    let meta_data = fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", meta_path.display()));
    let v: Value = serde_json::from_str(&meta_data)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", meta_path.display()));
    let fmt_val = v.get("format").and_then(|x| x.as_str()).unwrap_or_else(|| {
        panic!(
            "format key missing or not a string in {}",
            meta_path.display()
        )
    });
    assert_eq!(fmt_val, format, "export.meta.json format mismatch");
}
