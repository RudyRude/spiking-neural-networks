#![allow(clippy::needless_collect)]
//! NDF-H Core: membership ledger and minimal hypergraph snapshot types.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Minimal vertex identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NeuronId(u32);
impl From<u32> for NeuronId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl NeuronId {
    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Minimal hyperedge identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HyperedgeId(u32);
impl From<u32> for HyperedgeId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl HyperedgeId {
    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Hyperedge arity semantics (kept for compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperedgeType {
    ManyToOne,
    // Future variants could be added if needed
}

/// Minimal hyperedge structure: sources (tails) -> targets (heads)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperedge {
    id: HyperedgeId,
    pub sources: Vec<NeuronId>,
    pub targets: Vec<NeuronId>,
    _kind: HyperedgeType,
}

impl Hyperedge {
    pub fn new(
        id: HyperedgeId,
        sources: Vec<NeuronId>,
        targets: Vec<NeuronId>,
        kind: HyperedgeType,
    ) -> Result<Self, &'static str> {
        if sources.is_empty() || targets.is_empty() {
            return Err("empty endpoint set");
        }
        Ok(Self {
            id,
            sources,
            targets,
            _kind: kind,
        })
    }

    pub fn id(&self) -> HyperedgeId {
        self.id
    }
}

/// Minimal in-memory hypergraph network used by exporters and tests
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct HypergraphNetwork {
    edges: BTreeMap<HyperedgeId, Hyperedge>,
}

impl HypergraphNetwork {
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    pub fn add_hyperedge(&mut self, e: Hyperedge) -> Result<(), &'static str> {
        if self.edges.contains_key(&e.id) {
            return Err("duplicate hyperedge id");
        }
        self.edges.insert(e.id, e);
        Ok(())
    }

    pub fn get_hyperedge(&self, id: HyperedgeId) -> Option<&Hyperedge> {
        self.edges.get(&id)
    }

    pub fn hyperedge_ids(&self) -> Vec<HyperedgeId> {
        self.edges.keys().copied().collect()
    }
}

/// Valid-time membership row (tail membership into hyperedge h_id)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MembershipRow {
    pub h_id: u64,
    pub tail_v: u64,
    pub t_start: i64,
    pub t_end: Option<i64>,
}

/// Append-only membership log
#[derive(Debug, Default)]
pub struct MembershipLog {
    rows: Vec<MembershipRow>,
}

impl MembershipLog {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append a new membership (open-ended)
    pub fn add(&mut self, h_id: u64, tail_v: u64, t_start: i64) {
        self.rows.push(MembershipRow {
            h_id,
            tail_v,
            t_start,
            t_end: None,
        });
    }

    /// Close an existing membership by setting its t_end
    pub fn remove(&mut self, h_id: u64, tail_v: u64, t_end: i64) {
        if let Some(row) = self
            .rows
            .iter_mut()
            .rev()
            .find(|r| r.h_id == h_id && r.tail_v == tail_v && r.t_end.is_none())
        {
            row.t_end = Some(t_end);
        } else {
            // In a full implementation, record invariant violation (missing open membership)
        }
    }

    /// Build a snapshot hypergraph "AS OF" event time t_ns (empty baseline).
    /// Prefer `snapshot_as_of_with_catalog` to materialize directed hyperedges.
    pub fn snapshot_as_of(&self, _t_ns: i64) -> HypergraphNetwork {
        HypergraphNetwork::new()
    }

    /// Build a snapshot HypergraphNetwork at time t_ns using a catalog that maps h_id -> head_v.
    /// Each hyperedge becomes ManyToOne: sources = active tails at t_ns, target = head_v.
    pub fn snapshot_as_of_with_catalog(
        &self,
        t_ns: i64,
        catalog: &HyperedgeCatalog,
    ) -> HypergraphNetwork {
        // 1) Collect active memberships at t_ns grouped by h_id
        let mut tails_by_h: HashMap<u64, Vec<u64>> = HashMap::new();
        for row in self.rows.iter() {
            let active = row.t_start <= t_ns && row.t_end.map(|e| e > t_ns).unwrap_or(true);
            if active {
                tails_by_h.entry(row.h_id).or_default().push(row.tail_v);
            }
        }

        // 2) Materialize hyperedges
        let mut net = HypergraphNetwork::new();
        for (h_id_u64, tails) in tails_by_h.into_iter() {
            if let Some(&head_v_u64) = catalog.head_map.get(&h_id_u64) {
                let hed_id = HyperedgeId::from(h_id_u64 as u32);
                let head = NeuronId::from(head_v_u64 as u32);
                let sources: Vec<NeuronId> = tails
                    .into_iter()
                    .map(|v| NeuronId::from(v as u32))
                    .collect();

                if sources.is_empty() {
                    continue;
                }

                if let Ok(edge) =
                    Hyperedge::new(hed_id, sources, vec![head], HyperedgeType::ManyToOne)
                {
                    let _ = net.add_hyperedge(edge);
                }
            }
        }

        net
    }

    pub fn iter(&self) -> impl Iterator<Item = &MembershipRow> {
        self.rows.iter()
    }
}

/// Catalog of hyperedges providing head vertex mapping (h_id -> head_v)
#[derive(Debug, Default)]
pub struct HyperedgeCatalog {
    pub(crate) head_map: HashMap<u64, u64>,
}

impl HyperedgeCatalog {
    pub fn new() -> Self {
        Self {
            head_map: HashMap::new(),
        }
    }

    /// Register a hyperedge head mapping
    pub fn register_head(&mut self, h_id: u64, head_v: u64) {
        self.head_map.insert(h_id, head_v);
    }

    /// Bulk register heads
    pub fn extend_heads<I: IntoIterator<Item = (u64, u64)>>(&mut self, iter: I) {
        self.head_map.extend(iter);
    }

    /// Lookup head
    pub fn head_of(&self, h_id: u64) -> Option<u64> {
        self.head_map.get(&h_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_only_add_rem() {
        let mut log = MembershipLog::new();
        log.add(1, 2001, 100);
        log.remove(1, 2001, 200);
        assert_eq!(log.rows.len(), 1);
        assert_eq!(log.rows[0].t_end, Some(200));
    }

    #[test]
    fn snapshot_builds_many_to_one_edges() {
        let mut log = MembershipLog::new();
        // h_id=1, tails: 10,11; head: 99 active at t=150
        log.add(1, 10, 100);
        log.add(1, 11, 120);
        // a tail removed before t=150 should not appear
        log.add(1, 12, 90);
        log.remove(1, 12, 110);

        let mut cat = HyperedgeCatalog::new();
        cat.register_head(1, 99);

        let net = log.snapshot_as_of_with_catalog(150, &cat);
        // One hyperedge expected with id 1
        let ids = net.hyperedge_ids();
        assert!(ids.iter().any(|h| h.raw() == 1));
        // Validate sources/targets shape
        let edge = net.get_hyperedge(HyperedgeId::from(1)).unwrap();
        assert_eq!(edge.targets.len(), 1);
        assert!(edge.sources.len() >= 2); // 10 and 11 present
    }
}
