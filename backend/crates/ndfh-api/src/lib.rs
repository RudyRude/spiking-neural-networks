#![allow(dead_code)]
//! NDF-H API skeleton: procedural operations interface (no transport yet).
pub mod observability;

use ndfh_core::{HyperedgeCatalog, HypergraphNetwork, MembershipLog};
use ndfh_hgts::AsOfEngine;
use tracing::{info, instrument};

#[derive(Debug)]
pub struct HeCreate {
    pub head_v: u64,
    pub fe_spec_json: String,
    pub state_schema_json: Option<String>,
}

/// Transactional API for topology/state mutations and logs
pub trait TxnApi {
    fn he_create(&mut self, req: HeCreate) -> anyhow::Result<u64>;
    fn he_retire(&mut self, h_id: u64) -> anyhow::Result<()>;
    fn mem_add(&mut self, h_id: u64, tail_v: u64, t_start: i64) -> anyhow::Result<()>;
    fn mem_rem(&mut self, h_id: u64, tail_v: u64, t_end: i64) -> anyhow::Result<()>;
    fn state_upd(&mut self, h_id: u64, op: &str, payload_json: &str) -> anyhow::Result<()>;
    fn fire_append(&mut self, h_id: u64, t_ns: i64, payload_bytes: &[u8]) -> anyhow::Result<()>;
}

/// In-memory implementation suitable for tests and prototyping
pub struct InMemoryTxn {
    pub membership: MembershipLog,
    pub catalog: HyperedgeCatalog,
    next_h_id: u64,
}

impl Default for InMemoryTxn {
    fn default() -> Self {
        Self {
            membership: MembershipLog::new(),
            catalog: HyperedgeCatalog::new(),
            next_h_id: 1,
        }
    }
}

impl InMemoryTxn {
    /// Build a HypergraphNetwork snapshot (AS OF t_ns)
    #[instrument(skip(self))]
    pub fn snapshot_as_of(&self, t_ns: i64) -> HypergraphNetwork {
        info!(t_ns, "creating snapshot");
        AsOfEngine::snapshot_with_catalog(&self.membership, &self.catalog, t_ns)
    }
}

impl TxnApi for InMemoryTxn {
    fn he_create(&mut self, req: HeCreate) -> anyhow::Result<u64> {
        info!("creating hyperedge");
        let h_id = self.next_h_id;
        self.next_h_id += 1;

        // Register head mapping in catalog; fe_spec/state_schema would be persisted in a full impl
        self.catalog.register_head(h_id, req.head_v);
        Ok(h_id)
    }

    fn he_retire(&mut self, _h_id: u64) -> anyhow::Result<()> {
        info!(h_id = _h_id, "retiring hyperedge");
        // For now: no-op (retire would affect a hyperedges table in storage)
        Ok(())
    }

    fn mem_add(&mut self, h_id: u64, tail_v: u64, t_start: i64) -> anyhow::Result<()> {
        info!(h_id, tail_v, t_start, "adding membership");
        self.membership.add(h_id, tail_v, t_start);
        Ok(())
    }

    fn mem_rem(&mut self, h_id: u64, tail_v: u64, t_end: i64) -> anyhow::Result<()> {
        info!(h_id, tail_v, t_end, "removing membership");
        self.membership.remove(h_id, tail_v, t_end);
        Ok(())
    }

    fn state_upd(&mut self, _h_id: u64, _op: &str, _payload_json: &str) -> anyhow::Result<()> {
        info!(h_id = _h_id, op = _op, "updating state");
        // Validate against state_schema in a full implementation
        Ok(())
    }

    fn fire_append(&mut self, _h_id: u64, _t_ns: i64, _payload_bytes: &[u8]) -> anyhow::Result<()> {
        info!(h_id = _h_id, t_ns = _t_ns, "appending fire event");
        // Append to fire log in a full implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_of_snapshot_builds_edges_via_api() {
        let mut txn = InMemoryTxn::default();

        // Create hyperedge with head_v=99
        let h_id = txn
            .he_create(HeCreate {
                head_v: 99,
                fe_spec_json: "{}".to_string(),
                state_schema_json: None,
            })
            .expect("he_create");

        // Memberships: tails 10, 11 active at t=150; tail 12 removed before 150
        txn.mem_add(h_id, 10, 100).expect("mem_add 10");
        txn.mem_add(h_id, 11, 120).expect("mem_add 11");
        txn.mem_add(h_id, 12, 90).expect("mem_add 12");
        txn.mem_rem(h_id, 12, 110).expect("mem_rem 12");

        // Build snapshot
        let net = txn.snapshot_as_of(150);

        // Validate hyperedge presence and shape
        let ids = net.hyperedge_ids();
        assert!(
            ids.iter().any(|id| id.raw() == (h_id as u32)),
            "expected hyperedge id {} in snapshot",
            h_id
        );

        let edge = net
            .get_hyperedge(ndfh_core::HyperedgeId::from(h_id as u32))
            .expect("edge present");
        assert_eq!(edge.targets.len(), 1, "ManyToOne target cardinality");
        assert!(
            edge.sources.len() >= 2,
            "expected at least sources 10 and 11 active at t=150; got {}",
            edge.sources.len()
        );
    }
}
