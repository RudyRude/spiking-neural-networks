#![allow(dead_code)]
//! NDF-H HGTS: temporal semantics engine (AS OF / OVER) — skeleton.

use ndfh_core::{HyperedgeCatalog, HypergraphNetwork, MembershipLog};

#[derive(Debug, Clone, Copy)]
pub enum TemporalContext {
    AsOf(i64),
    Over { start: i64, end: i64 },
}

pub struct AsOfEngine;

impl AsOfEngine {
    /// Snapshot without head catalog (returns empty network baseline)
    pub fn snapshot(log: &MembershipLog, t_ns: i64) -> HypergraphNetwork {
        log.snapshot_as_of(t_ns)
    }

    /// Snapshot with head catalog (materializes directed hyperedges)
    pub fn snapshot_with_catalog(
        log: &MembershipLog,
        catalog: &HyperedgeCatalog,
        t_ns: i64,
    ) -> HypergraphNetwork {
        log.snapshot_as_of_with_catalog(t_ns, catalog)
    }
}
