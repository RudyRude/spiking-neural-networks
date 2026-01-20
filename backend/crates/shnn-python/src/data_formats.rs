//! Python bindings for NDF-H data formats and export functionality
//!
//! This module provides Python interfaces for exporting spike data and network
//! configurations in standardized NDF-H formats like GraphML, LPG-JSON, and RDF-NQuads.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyRuntimeError;

use ndfh_core::{HypergraphNetwork as NdfHypergraphNetwork, Hyperedge, HyperedgeId, HyperedgeType, NeuronId};

/// Python wrapper for NDF-H HypergraphNetwork
#[pyclass(name = "NDFHypergraph")]
pub struct PyNDFHypergraph {
    inner: NdfHypergraphNetwork,
}

#[pymethods]
impl PyNDFHypergraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: NdfHypergraphNetwork::new(),
        }
    }

    /// Add a hyperedge
    #[pyo3(signature = (id, sources, targets))]
    fn add_hyperedge(&mut self, id: u32, sources: Vec<u32>, targets: Vec<u32>) -> PyResult<()> {
        let hid = HyperedgeId::from(id);
        let src_ids: Vec<NeuronId> = sources.into_iter().map(NeuronId::from).collect();
        let tgt_ids: Vec<NeuronId> = targets.into_iter().map(NeuronId::from).collect();

        let edge = Hyperedge::new(hid, src_ids, tgt_ids, HyperedgeType::ManyToOne)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create hyperedge: {}", e)))?;

        self.inner.add_hyperedge(edge)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add hyperedge: {}", e)))
    }

    /// Get hyperedge IDs
    fn hyperedge_ids(&self) -> Vec<u32> {
        self.inner.hyperedge_ids().into_iter().map(|id| id.raw()).collect()
    }

    /// Get hyperedge by ID
    fn get_hyperedge(&self, id: u32) -> PyResult<PyObject> {
        if let Some(edge) = self.inner.get_hyperedge(HyperedgeId::from(id)) {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("id", edge.id().raw())?;
                dict.set_item("sources", edge.sources.iter().map(|n| n.raw()).collect::<Vec<_>>())?;
                dict.set_item("targets", edge.targets.iter().map(|n| n.raw()).collect::<Vec<_>>())?;
                Ok(dict.to_object(py))
            })
        } else {
            Err(PyRuntimeError::new_err(format!("Hyperedge {} not found", id)))
        }
    }

    fn __repr__(&self) -> String {
        format!("NDFHypergraph(edges={})", self.inner.hyperedge_ids().len())
    }
}

/// Python wrapper for data format exporters
#[pyclass(name = "DataFormatExporter")]
pub struct PyDataFormatExporter;

#[pymethods]
impl PyDataFormatExporter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Export spike data to GraphML format
    fn spikes_to_graphml(&self, spikes: Vec<(u64, u32)>) -> String {
        let mut graphml = String::new();
        graphml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        graphml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");
        graphml.push_str("  <graph id=\"spikes\" edgedefault=\"directed\">\n");

        // Add nodes for neurons
        let mut neurons = std::collections::HashSet::new();
        for (_, neuron_id) in &spikes {
            neurons.insert(*neuron_id);
        }
        for neuron_id in neurons {
            graphml.push_str(&format!("    <node id=\"neuron_{}\"/>\n", neuron_id));
        }

        // Add edges for spikes
        for (i, (time_ns, neuron_id)) in spikes.iter().enumerate() {
            graphml.push_str(&format!("    <edge id=\"spike_{}\" source=\"neuron_{}\" target=\"spike_event_{}\">\n", i, neuron_id, i));
            graphml.push_str(&format!("      <data key=\"time_ns\">{}</data>\n", time_ns));
            graphml.push_str("    </edge>\n");
            graphml.push_str(&format!("    <node id=\"spike_event_{}\">\n", i));
            graphml.push_str(&format!("      <data key=\"type\">spike</data>\n"));
            graphml.push_str(&format!("      <data key=\"time_ns\">{}</data>\n", time_ns));
            graphml.push_str("    </node>\n");
        }

        graphml.push_str("  </graph>\n");
        graphml.push_str("</graphml>\n");
        graphml
    }

    /// Export spike data to LPG-JSON format
    fn spikes_to_lpg_json(&self, spikes: Vec<(u64, u32)>) -> PyResult<String> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 0;

        // Neurons
        let mut neuron_map = std::collections::HashMap::new();
        for (_, neuron_id) in &spikes {
            if !neuron_map.contains_key(neuron_id) {
                neuron_map.insert(*neuron_id, node_id);
                nodes.push(serde_json::json!({
                    "id": node_id,
                    "labels": ["Neuron"],
                    "properties": {
                        "neuron_id": neuron_id
                    }
                }));
                node_id += 1;
            }
        }

        // Spike events
        for (time_ns, neuron_id) in spikes {
            let spike_id = node_id;
            nodes.push(serde_json::json!({
                "id": spike_id,
                "labels": ["Spike"],
                "properties": {
                    "time_ns": time_ns
                }
            }));
            edges.push(serde_json::json!({
                "id": node_id + 1,
                "type": "SPIKED",
                "start": neuron_map[&neuron_id],
                "end": spike_id,
                "properties": {}
            }));
            node_id += 2;
        }

        let lpg = serde_json::json!({
            "nodes": nodes,
            "edges": edges
        });

        Ok(serde_json::to_string_pretty(&lpg)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {}", e)))?)
    }

    /// Export spike data to RDF N-Quads format
    fn spikes_to_rdf_nquads(&self, spikes: Vec<(u64, u32)>) -> String {
        let mut nquads = String::new();

        for (time_ns, neuron_id) in spikes {
            nquads.push_str(&format!("<http://example.org/neuron/{}> <http://example.org/has_spike> <http://example.org/spike/{}_{}> .\n", neuron_id, neuron_id, time_ns));
            nquads.push_str(&format!("<http://example.org/spike/{}_{}> <http://example.org/time_ns> \"{}\" .\n", neuron_id, time_ns, time_ns));
        }

        nquads
    }

    /// Export network connectivity to GraphML format
    fn network_to_graphml(&self, connections: Vec<(u32, u32, f32)>) -> String {
        let mut graphml = String::new();
        graphml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        graphml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");
        graphml.push_str("  <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"float\"/>\n");
        graphml.push_str("  <graph id=\"network\" edgedefault=\"directed\">\n");

        // Add nodes for neurons
        let mut neurons = std::collections::HashSet::new();
        for (source, target, _) in &connections {
            neurons.insert(*source);
            neurons.insert(*target);
        }
        for neuron_id in neurons {
            graphml.push_str(&format!("    <node id=\"n{}\"/>\n", neuron_id));
        }

        // Add edges for connections
        for (i, (source, target, weight)) in connections.iter().enumerate() {
            graphml.push_str(&format!("    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">\n", i, source, target));
            graphml.push_str(&format!("      <data key=\"weight\">{}</data>\n", weight));
            graphml.push_str("    </edge>\n");
        }

        graphml.push_str("  </graph>\n");
        graphml.push_str("</graphml>\n");
        graphml
    }

    /// Export network connectivity to LPG-JSON format
    fn network_to_lpg_json(&self, connections: Vec<(u32, u32, f32)>) -> PyResult<String> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 0;

        // Neurons
        let mut neuron_map = std::collections::HashMap::new();
        for (source, target, _) in &connections {
            for &neuron_id in &[*source, *target] {
                if !neuron_map.contains_key(&neuron_id) {
                    neuron_map.insert(neuron_id, node_id);
                    nodes.push(serde_json::json!({
                        "id": node_id,
                        "labels": ["Neuron"],
                        "properties": {
                            "neuron_id": neuron_id
                        }
                    }));
                    node_id += 1;
                }
            }
        }

        // Connections
        for (source, target, weight) in connections {
            edges.push(serde_json::json!({
                "id": node_id,
                "type": "CONNECTS_TO",
                "start": neuron_map[&source],
                "end": neuron_map[&target],
                "properties": {
                    "weight": weight
                }
            }));
            node_id += 1;
        }

        let lpg = serde_json::json!({
            "nodes": nodes,
            "edges": edges
        });

        Ok(serde_json::to_string_pretty(&lpg)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {}", e)))?)
    }

    /// Export network connectivity to RDF N-Quads format
    fn network_to_rdf_nquads(&self, connections: Vec<(u32, u32, f32)>) -> String {
        let mut nquads = String::new();

        for (source, target, weight) in connections {
            nquads.push_str(&format!("<http://example.org/neuron/{}> <http://example.org/connects_to> <http://example.org/neuron/{}> .\n", source, target));
            nquads.push_str(&format!("<http://example.org/connection/{}_{}> <http://example.org/weight> \"{}\" .\n", source, target, weight));
        }

        nquads
    }
}

/// Python wrapper for format conversion utilities
#[pyclass(name = "FormatConverter")]
pub struct PyFormatConverter;

#[pymethods]
impl PyFormatConverter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Convert spike times to binary array
    fn spike_times_to_binary(&self, spike_times: Vec<f64>, duration: f64, resolution: f64) -> Vec<u8> {
        let bins = (duration / resolution) as usize;
        let mut binary_array = vec![0u8; bins];

        for spike_time in spike_times {
            let bin = (spike_time / resolution) as usize;
            if bin < bins {
                binary_array[bin] = 1;
            }
        }

        binary_array
    }

    /// Convert binary array to spike times
    fn binary_to_spike_times(&self, binary_array: Vec<u8>, resolution: f64) -> Vec<f64> {
        let mut spike_times = Vec::new();

        for (i, &bit) in binary_array.iter().enumerate() {
            if bit != 0 {
                spike_times.push(i as f64 * resolution);
            }
        }

        spike_times
    }

    /// Convert spike trains to raster matrix
    fn spike_trains_to_raster(&self, spike_trains: Vec<Vec<f64>>, duration: f64, resolution: f64) -> PyResult<PyObject> {
        let n_neurons = spike_trains.len();
        let n_bins = (duration / resolution) as usize;

        Python::with_gil(|py| {
            let matrix = PyList::empty(py);

            for spike_train in spike_trains {
                let binary = self.spike_times_to_binary(spike_train, duration, resolution);
                matrix.append(binary)?;
            }

            Ok(matrix.to_object(py))
        })
    }

    /// Convert raster matrix to spike trains
    fn raster_to_spike_trains(&self, raster_matrix: Vec<Vec<u8>>, resolution: f64) -> Vec<Vec<f64>> {
        raster_matrix.into_iter()
            .map(|binary_array| self.binary_to_spike_times(binary_array, resolution))
            .collect()
    }
}