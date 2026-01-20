//! Python bindings for advanced neural connectivity structures
//!
//! This module provides Python classes and functions for creating and managing
//! different types of neural connectivity: hypergraphs, graphs, dense matrices,
//! and sparse matrices.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use shnn_core::connectivity::{
    NetworkConnectivity, BatchConnectivity, PlasticConnectivity, WeightSnapshotConnectivity,
    HypergraphNetwork, GraphNetwork, MatrixNetwork, SparseMatrixNetwork,
    ConnectivityStats, PlasticConn
};
use shnn_core::{NeuronId, Spike, SpikeRoute, Time};

use crate::error_conversion::ffi_error_to_py_err;

/// Python wrapper for HypergraphNetwork
#[pyclass(name = "HypergraphNetwork")]
pub struct PyHypergraphNetwork {
    inner: HypergraphNetwork,
}

#[pymethods]
impl PyHypergraphNetwork {
    #[new]
    fn new() -> Self {
        Self {
            inner: HypergraphNetwork::new(),
        }
    }

    /// Add a hyperedge connecting multiple pre-synaptic neurons to multiple post-synaptic neurons
    #[pyo3(signature = (pre_neurons, post_neurons, weight=1.0, delay=None))]
    fn add_hyperedge(
        &mut self,
        pre_neurons: Vec<u32>,
        post_neurons: Vec<u32>,
        weight: Option<f32>,
        delay: Option<f64>,
    ) -> PyResult<()> {
        let pre_ids: Vec<NeuronId> = pre_neurons.into_iter().map(NeuronId::new).collect();
        let post_ids: Vec<NeuronId> = post_neurons.into_iter().map(NeuronId::new).collect();
        let weight = weight.unwrap_or(1.0);
        let delay = delay.map(|d| Time::from_millis(d));

        self.inner.add_hyperedge(&pre_ids, &post_ids, weight, delay)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add hyperedge: {:?}", e)))
    }

    /// Route a spike through the hypergraph
    fn route_spike(&self, neuron_id: u32, time: f64) -> PyResult<PyObject> {
        let spike = Spike::new(NeuronId::new(neuron_id), Time::from_millis(time), 1.0);
        let current_time = Time::from_millis(time);

        Python::with_gil(|py| {
            match self.inner.route_spike(&spike, current_time) {
                Ok(routes) => {
                    let list = PyList::empty(py);
                    for route in routes {
                        let route_dict = pyo3::types::PyDict::new(py);
                        route_dict.set_item("source_connection", route.source_connection)?;
                        route_dict.set_item("targets", route.targets.iter().map(|id| id.raw()).collect::<Vec<_>>())?;
                        route_dict.set_item("weights", &route.weights)?;
                        route_dict.set_item("delays", route.delays.iter().map(|d| d.as_millis()).collect::<Vec<_>>())?;
                        list.append(route_dict)?;
                    }
                    Ok(list.to_object(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Spike routing failed: {:?}", e))),
            }
        })
    }

    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("connection_count", stats.connection_count)?;
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("sparsity", stats.sparsity)?;
            dict.set_item("avg_degree", stats.avg_degree)?;
            dict.set_item("max_degree", stats.max_degree)?;
            Ok(dict.to_object(py))
        })
    }

    /// Get all neuron IDs in the network
    fn neurons(&self) -> Vec<u32> {
        self.inner.neurons().into_iter().map(|id| id.raw()).collect()
    }

    /// Get connection count
    fn connection_count(&self) -> usize {
        self.inner.connection_count()
    }

    /// Check if network is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("HypergraphNetwork(neurons={}, connections={})",
                self.inner.neuron_count(), self.inner.connection_count())
    }
}

/// Python wrapper for GraphNetwork
#[pyclass(name = "GraphNetwork")]
pub struct PyGraphNetwork {
    inner: GraphNetwork,
}

#[pymethods]
impl PyGraphNetwork {
    #[new]
    fn new() -> Self {
        Self {
            inner: GraphNetwork::new(),
        }
    }

    /// Add a directed connection between two neurons
    #[pyo3(signature = (source, target, weight=1.0, delay=None))]
    fn add_connection(
        &mut self,
        source: u32,
        target: u32,
        weight: Option<f32>,
        delay: Option<f64>,
    ) -> PyResult<()> {
        let source_id = NeuronId::new(source);
        let target_id = NeuronId::new(target);
        let weight = weight.unwrap_or(1.0);
        let delay = delay.map(|d| Time::from_millis(d));

        self.inner.add_edge(source_id, target_id, weight, delay)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add connection: {:?}", e)))
    }

    /// Route a spike through the graph
    fn route_spike(&self, neuron_id: u32, time: f64) -> PyResult<PyObject> {
        let spike = Spike::new(NeuronId::new(neuron_id), Time::from_millis(time), 1.0);
        let current_time = Time::from_millis(time);

        Python::with_gil(|py| {
            match self.inner.route_spike(&spike, current_time) {
                Ok(routes) => {
                    let list = PyList::empty(py);
                    for route in routes {
                        let route_dict = pyo3::types::PyDict::new(py);
                        route_dict.set_item("source_connection", route.source_connection)?;
                        route_dict.set_item("targets", route.targets.iter().map(|id| id.raw()).collect::<Vec<_>>())?;
                        route_dict.set_item("weights", &route.weights)?;
                        route_dict.set_item("delays", route.delays.iter().map(|d| d.as_millis()).collect::<Vec<_>>())?;
                        list.append(route_dict)?;
                    }
                    Ok(list.to_object(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Spike routing failed: {:?}", e))),
            }
        })
    }

    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("connection_count", stats.connection_count)?;
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("sparsity", stats.sparsity)?;
            dict.set_item("avg_degree", stats.avg_degree)?;
            dict.set_item("max_degree", stats.max_degree)?;
            Ok(dict.to_object(py))
        })
    }

    /// Get all neuron IDs
    fn neurons(&self) -> Vec<u32> {
        self.inner.neurons().into_iter().map(|id| id.raw()).collect()
    }

    /// Get connection count
    fn connection_count(&self) -> usize {
        self.inner.connection_count()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("GraphNetwork(neurons={}, connections={})",
                self.inner.neuron_count(), self.inner.connection_count())
    }
}

/// Python wrapper for MatrixNetwork
#[pyclass(name = "MatrixNetwork")]
pub struct PyMatrixNetwork {
    inner: MatrixNetwork,
}

#[pymethods]
impl PyMatrixNetwork {
    #[new]
    #[pyo3(signature = (max_neurons, with_delays=false))]
    fn new(max_neurons: usize, with_delays: Option<bool>) -> Self {
        let inner = if with_delays.unwrap_or(false) {
            MatrixNetwork::with_delays(max_neurons)
        } else {
            MatrixNetwork::new(max_neurons)
        };
        Self { inner }
    }

    /// Set weight between neurons
    #[pyo3(signature = (source, target, weight, delay=None))]
    fn set_weight(
        &mut self,
        source: u32,
        target: u32,
        weight: f32,
        delay: Option<f64>,
    ) -> PyResult<()> {
        let source_id = NeuronId::new(source);
        let target_id = NeuronId::new(target);

        if let Some(delay_ms) = delay {
            self.inner.set_delay(source_id, target_id, Time::from_millis(delay_ms))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set delay: {:?}", e)))?;
        }

        self.inner.set_weight(source_id, target_id, weight)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set weight: {:?}", e)))
    }

    /// Get weight between neurons
    fn get_weight(&self, source: u32, target: u32) -> PyResult<f32> {
        self.inner.get_weight(NeuronId::new(source), NeuronId::new(target))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get weight: {:?}", e)))
    }

    /// Route a spike
    fn route_spike(&self, neuron_id: u32, time: f64) -> PyResult<PyObject> {
        let spike = Spike::new(NeuronId::new(neuron_id), Time::from_millis(time), 1.0);
        let current_time = Time::from_millis(time);

        Python::with_gil(|py| {
            match self.inner.route_spike(&spike, current_time) {
                Ok(routes) => {
                    let list = PyList::empty(py);
                    for route in routes {
                        let route_dict = pyo3::types::PyDict::new(py);
                        route_dict.set_item("source_connection", route.source_connection)?;
                        route_dict.set_item("targets", route.targets.iter().map(|id| id.raw()).collect::<Vec<_>>())?;
                        route_dict.set_item("weights", &route.weights)?;
                        route_dict.set_item("delays", route.delays.iter().map(|d| d.as_millis()).collect::<Vec<_>>())?;
                        list.append(route_dict)?;
                    }
                    Ok(list.to_object(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Spike routing failed: {:?}", e))),
            }
        })
    }

    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("connection_count", stats.connection_count)?;
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("sparsity", stats.sparsity)?;
            dict.set_item("avg_degree", stats.avg_degree)?;
            dict.set_item("max_degree", stats.max_degree)?;
            Ok(dict.to_object(py))
        })
    }

    /// Get capacity
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if full
    fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    fn __repr__(&self) -> String {
        format!("MatrixNetwork(capacity={}, neurons={}, connections={})",
                self.inner.capacity(), self.inner.neuron_count(), self.inner.connection_count())
    }
}

/// Python wrapper for SparseMatrixNetwork
#[pyclass(name = "SparseMatrixNetwork")]
pub struct PySparseMatrixNetwork {
    inner: SparseMatrixNetwork,
}

#[pymethods]
impl PySparseMatrixNetwork {
    #[new]
    #[pyo3(signature = (max_neurons, estimated_connections=None))]
    fn new(max_neurons: usize, estimated_connections: Option<usize>) -> Self {
        let inner = if let Some(capacity) = estimated_connections {
            SparseMatrixNetwork::with_capacity(max_neurons, capacity)
        } else {
            SparseMatrixNetwork::new(max_neurons)
        };
        Self { inner }
    }

    /// Set weight between neurons
    #[pyo3(signature = (source, target, weight, delay=None))]
    fn set_weight(
        &mut self,
        source: u32,
        target: u32,
        weight: f32,
        delay: Option<f64>,
    ) -> PyResult<()> {
        let source_id = NeuronId::new(source);
        let target_id = NeuronId::new(target);

        if let Some(delay_ms) = delay {
            // Note: SparseMatrixNetwork might not support delays directly
            // This would need to be extended if required
            return Err(PyRuntimeError::new_err("Delays not yet supported for sparse matrices"));
        }

        self.inner.set_weight(source_id, target_id, weight)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set weight: {:?}", e)))
    }

    /// Get weight between neurons
    fn get_weight(&self, source: u32, target: u32) -> PyResult<f32> {
        self.inner.get_weight(NeuronId::new(source), NeuronId::new(target))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get weight: {:?}", e)))
    }

    /// Route a spike
    fn route_spike(&self, neuron_id: u32, time: f64) -> PyResult<PyObject> {
        let spike = Spike::new(NeuronId::new(neuron_id), Time::from_millis(time), 1.0);
        let current_time = Time::from_millis(time);

        Python::with_gil(|py| {
            match self.inner.route_spike(&spike, current_time) {
                Ok(routes) => {
                    let list = PyList::empty(py);
                    for route in routes {
                        let route_dict = pyo3::types::PyDict::new(py);
                        route_dict.set_item("source_connection", route.source_connection)?;
                        route_dict.set_item("targets", route.targets.iter().map(|id| id.raw()).collect::<Vec<_>>())?;
                        route_dict.set_item("weights", &route.weights)?;
                        route_dict.set_item("delays", route.delays.iter().map(|d| d.as_millis()).collect::<Vec<_>>())?;
                        list.append(route_dict)?;
                    }
                    Ok(list.to_object(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Spike routing failed: {:?}", e))),
            }
        })
    }

    /// Get sparsity ratio
    fn sparsity(&self) -> f32 {
        self.inner.sparsity()
    }

    /// Get number of non-zero entries
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("connection_count", stats.connection_count)?;
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("sparsity", stats.sparsity)?;
            dict.set_item("avg_degree", stats.avg_degree)?;
            dict.set_item("max_degree", stats.max_degree)?;
            Ok(dict.to_object(py))
        })
    }

    fn __repr__(&self) -> String {
        format!("SparseMatrixNetwork(capacity={}, nnz={}, sparsity={:.3})",
                self.inner.capacity(), self.inner.nnz(), self.inner.sparsity())
    }
}

/// Python wrapper for PlasticConn enum
#[pyclass(name = "PlasticConnectivity")]
pub struct PyPlasticConnectivity {
    inner: PlasticConn,
}

#[pymethods]
impl PyPlasticConnectivity {
    #[classmethod]
    fn from_graph(_cls: &PyType, graph: PyRef<PyGraphNetwork>) -> Self {
        Self {
            inner: PlasticConn::from_graph(graph.inner.clone()),
        }
    }

    #[classmethod]
    fn from_matrix(_cls: &PyType, matrix: PyRef<PyMatrixNetwork>) -> Self {
        Self {
            inner: PlasticConn::from_matrix(matrix.inner.clone()),
        }
    }

    #[classmethod]
    fn from_sparse(_cls: &PyType, sparse: PyRef<PySparseMatrixNetwork>) -> Self {
        Self {
            inner: PlasticConn::from_sparse(sparse.inner.clone()),
        }
    }

    /// Route a spike
    fn route_spike(&self, neuron_id: u32, time: f64) -> PyResult<PyObject> {
        let spike = Spike::new(NeuronId::new(neuron_id), Time::from_millis(time), 1.0);
        let current_time = Time::from_millis(time);

        Python::with_gil(|py| {
            match self.inner.route_spike(&spike, current_time) {
                Ok(routes) => {
                    let list = PyList::empty(py);
                    for route in routes {
                        let route_dict = pyo3::types::PyDict::new(py);
                        route_dict.set_item("source_connection", route.source_connection)?;
                        route_dict.set_item("targets", route.targets.iter().map(|id| id.raw()).collect::<Vec<_>>())?;
                        route_dict.set_item("weights", &route.weights)?;
                        route_dict.set_item("delays", route.delays.iter().map(|d| d.as_millis()).collect::<Vec<_>>())?;
                        list.append(route_dict)?;
                    }
                    Ok(list.to_object(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Spike routing failed: {:?}", e))),
            }
        })
    }

    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("connection_count", stats.connection_count)?;
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("sparsity", stats.sparsity)?;
            dict.set_item("avg_degree", stats.avg_degree)?;
            dict.set_item("max_degree", stats.max_degree)?;
            Ok(dict.to_object(py))
        })
    }

    fn __repr__(&self) -> String {
        format!("PlasticConnectivity(neurons={}, connections={})",
                self.inner.neuron_count(), self.inner.connection_count())
    }
}