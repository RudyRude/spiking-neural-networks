#![allow(dead_code)]
//! HDX I/O helpers for reading JSONL shards into core temporal structures.
//!
//! Minimal bootstrap to support:
//! - Loading membership append-only log (valid-time) from JSONL shards
//! - Loading hyperedge head catalog (h_id -> head_v) from JSONL shards
//!
//! These helpers are intentionally lightweight and schema-tolerant for early fixtures.

use crate::{DatasetManifest, HdxResult};
use ndfh_core::{HyperedgeCatalog, MembershipLog};
use serde_json::Value as JsonValue;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Resolve shard file paths for a given logical table name (e.g., "membership", "hyperedges")
pub fn resolve_table_shards(mf: &DatasetManifest, root: &Path, table: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for shard in mf.shards.values() {
        if shard.table.as_str() == table {
            files.push(root.join(&shard.path));
        }
    }
    files
}

/// Load MembershipLog from all membership shards in the manifest.
/// Returns Ok(Some(log)) when membership shards are present; Ok(None) if no membership shards.
pub fn load_membership_log_from_manifest(
    mf: &DatasetManifest,
    root: &Path,
) -> HdxResult<Option<MembershipLog>> {
    let mut files = resolve_table_shards(mf, root, "membership");
    // Fallback: conventional fixture filenames when manifest doesn't list membership shards
    if files.is_empty() {
        let fallback_candidates = [
            root.join("membership.sample.jsonl"),
            root.join("membership.jsonl"),
        ];
        for cand in fallback_candidates {
            if cand.exists() {
                files.push(cand);
            }
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    let mut log = MembershipLog::new();
    for file in files {
        if !file.exists() {
            continue;
        }
        load_membership_jsonl_file(&file, &mut log)?;
    }
    Ok(Some(log))
}

/// Parse one membership JSONL file and append rows to the log.
/// Expected fields per line: h_id: u64, tail_v: u64, t_start: i64, t_end: Option<i64>
fn load_membership_jsonl_file(path: &Path, log: &mut MembershipLog) -> HdxResult<()> {
    let f = fs::File::open(path)?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: JsonValue = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // skip malformed lines
        };

        let h_id = v.get("h_id").and_then(|x| x.as_u64());
        let tail_v = v.get("tail_v").and_then(|x| x.as_u64());
        let t_start = v.get("t_start").and_then(|x| x.as_i64());
        let t_end = match v.get("t_end") {
            Some(JsonValue::Null) | None => None,
            Some(x) => x.as_i64(),
        };

        match (h_id, tail_v, t_start) {
            (Some(h), Some(t), Some(ts)) => {
                log.add(h, t, ts);
                if let Some(te) = t_end {
                    log.remove(h, t, te);
                }
            }
            _ => {
                // skip row; could log a warning in a fuller impl
                continue;
            }
        }
    }
    Ok(())
}

/// Load HyperedgeCatalog from all hyperedges shards in the manifest.
/// Returns Ok(Some(catalog)) when hyperedges shards are present; Ok(None) if no hyperedges shards.
pub fn load_hyperedge_catalog_from_manifest(
    mf: &DatasetManifest,
    root: &Path,
) -> HdxResult<Option<HyperedgeCatalog>> {
    let mut files = resolve_table_shards(mf, root, "hyperedges");
    // Fallback: conventional fixture filenames when manifest doesn't list hyperedges shards
    if files.is_empty() {
        let fallback_candidates = [
            root.join("hyperedges.sample.jsonl"),
            root.join("hyperedges.jsonl"),
        ];
        for cand in fallback_candidates {
            if cand.exists() {
                files.push(cand);
            }
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    let mut cat = HyperedgeCatalog::new();
    for file in files {
        if !file.exists() {
            continue;
        }
        load_hyperedges_jsonl_file(&file, &mut cat)?;
    }
    Ok(Some(cat))
}

/// Parse one hyperedges JSONL file and register head mappings into the catalog.
/// Expected minimal fields per line: h_id: u64, head_v: u64
fn load_hyperedges_jsonl_file(path: &Path, cat: &mut HyperedgeCatalog) -> HdxResult<()> {
    let f = fs::File::open(path)?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: JsonValue = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // skip malformed lines
        };

        let h_id = v.get("h_id").and_then(|x| x.as_u64());
        let head_v = v.get("head_v").and_then(|x| x.as_u64());

        if let (Some(h), Some(head)) = (h_id, head_v) {
            cat.register_head(h, head);
        }
    }
    Ok(())
}
