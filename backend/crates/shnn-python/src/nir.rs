//! Python bindings for NIR (Neural Intermediate Representation) compilation
//!
//! This module provides Python interfaces for compiling NIR programs and
//! executing them through the CLI-first workflow.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use shnn_ir::{
    Module, parse_text,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
};
use shnn_compiler::{compile_with_passes, verify_module, list_ops};
use shnn_cli::commands::nir::{SpikesFormat, NirCompile, NirRun, NirVerify};
use shnn_storage::{vevt::{VEVTEvent, encode_vevt}, StreamId, Time as StorageTime};

use std::path::PathBuf;

/// Python wrapper for NIR operations
#[pyclass(name = "NIRCompiler")]
pub struct PyNIRCompiler;

#[pymethods]
impl PyNIRCompiler {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compile network configuration to NIR textual format
    #[pyo3(signature = (
        output_path,
        neurons="lif",
        plasticity="stdp",
        inputs=10,
        hidden=50,
        outputs=5,
        topology="fully-connected",
        steps=10000,
        dt_us=100,
        stimulus="poisson",
        stimulus_rate=20.0,
        record_potentials=false,
        seed=None
    ))]
    fn compile_to_file(
        &self,
        output_path: String,
        neurons: Option<String>,
        plasticity: Option<String>,
        inputs: Option<u32>,
        hidden: Option<u32>,
        outputs: Option<u32>,
        topology: Option<String>,
        steps: Option<u64>,
        dt_us: Option<u64>,
        stimulus: Option<String>,
        stimulus_rate: Option<f32>,
        record_potentials: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let args = NirCompile {
            output: PathBuf::from(output_path),
            neurons: neurons.unwrap_or("lif".to_string()).parse().unwrap_or_default(),
            plasticity: plasticity.unwrap_or("stdp".to_string()).parse().unwrap_or_default(),
            inputs: inputs.unwrap_or(10),
            hidden: hidden.unwrap_or(50),
            outputs: outputs.unwrap_or(5),
            topology: topology.unwrap_or("fully-connected".to_string()).parse().unwrap_or_default(),
            steps: steps.unwrap_or(10000),
            dt_us: dt_us.unwrap_or(100),
            stimulus: stimulus.unwrap_or("poisson".to_string()).parse().unwrap_or_default(),
            stimulus_rate: stimulus_rate.unwrap_or(20.0),
            record_potentials: record_potentials.unwrap_or(false),
            seed,
        };

        // This would need async runtime, but for now we'll simulate
        // In real implementation, this would call args.execute().await
        Err(PyRuntimeError::new_err("NIR compilation not yet implemented in Python bindings"))
    }

    /// Parse and run NIR program from file
    #[pyo3(signature = (nir_path, output_path=None, spikes_format="json"))]
    fn run_from_file(
        &self,
        nir_path: String,
        output_path: Option<String>,
        spikes_format: Option<String>,
    ) -> PyResult<PyObject> {
        let format = match spikes_format.as_deref().unwrap_or("json") {
            "json" => SpikesFormat::Json,
            "vevt" => SpikesFormat::Vevt,
            "graphml" => SpikesFormat::GraphML,
            "lpg-json" => SpikesFormat::LPGJson,
            "rdf-nquads" => SpikesFormat::RDFNQuads,
            _ => return Err(PyValueError::new_err("Unsupported spikes format")),
        };

        let args = NirRun {
            input: PathBuf::from(nir_path),
            output: output_path.map(PathBuf::from),
            spikes_format: format,
        };

        // This would need async runtime
        // In real implementation, this would call args.execute().await
        Err(PyRuntimeError::new_err("NIR execution not yet implemented in Python bindings"))
    }

    /// Verify NIR program from file
    fn verify_file(&self, nir_path: String) -> PyResult<()> {
        let args = NirVerify {
            input: PathBuf::from(nir_path),
        };

        // This would need async runtime
        // In real implementation, this would call args.execute().await
        Err(PyRuntimeError::new_err("NIR verification not yet implemented in Python bindings"))
    }

    /// List available NIR operations
    #[pyo3(signature = (detailed=false))]
    fn list_operations(&self, detailed: Option<bool>) -> PyResult<PyObject> {
        let ops = list_ops();

        Python::with_gil(|py| {
            let result = PyDict::new(py);

            // Group ops by dialect
            let mut dialects: std::collections::BTreeMap<&str, Vec<_>> = std::collections::BTreeMap::new();
            for op in ops {
                dialects.entry(op.dialect).or_default().push(op);
            }

            for (dialect_name, dialect_ops) in dialects {
                let dialect_dict = PyDict::new(py);
                for op in dialect_ops {
                    let op_info = if detailed.unwrap_or(false) {
                        // Detailed mode: show attributes with types and docs
                        let attrs_dict = PyDict::new(py);
                        for attr in &op.attrs {
                            let attr_info = PyDict::new(py);
                            attr_info.set_item("type", attr.kind.name())?;
                            attr_info.set_item("required", attr.required)?;
                            attr_info.set_item("doc", &attr.doc)?;
                            attrs_dict.set_item(&attr.name, attr_info)?;
                        }
                        let op_dict = PyDict::new(py);
                        op_dict.set_item("version", op.version)?;
                        op_dict.set_item("attributes", attrs_dict)?;
                        op_dict.to_object(py)
                    } else {
                        // Compact mode: just show signature
                        let attrs: Vec<String> = op.attrs.iter().map(|a| {
                            let required = if a.required { "" } else { "?" };
                            format!("{}{}: {}", a.name, required, a.kind.name())
                        }).collect();
                        format!("{}@v{} {{ {} }}", op.name, op.version, attrs.join(", ")).to_object(py)
                    };
                    dialect_dict.set_item(&op.name, op_info)?;
                }
                result.set_item(dialect_name, dialect_dict)?;
            }

            Ok(result.to_object(py))
        })
    }

    /// Create a basic NIR module programmatically
    #[pyo3(signature = (
        num_inputs=10,
        num_hidden=50,
        num_outputs=5,
        stimulus_rate=20.0,
        dt_ms=0.1,
        duration_ms=1000.0,
        record_potentials=false,
        seed=None
    ))]
    fn create_basic_module(
        &self,
        num_inputs: Option<u32>,
        num_hidden: Option<u32>,
        num_outputs: Option<u32>,
        stimulus_rate: Option<f32>,
        dt_ms: Option<f32>,
        duration_ms: Option<f64>,
        record_potentials: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let inputs = num_inputs.unwrap_or(10);
        let hidden = num_hidden.unwrap_or(50);
        let outputs = num_outputs.unwrap_or(5);
        let rate = stimulus_rate.unwrap_or(20.0);
        let dt = dt_ms.unwrap_or(0.1);
        let duration = duration_ms.unwrap_or(1000.0);
        let record = record_potentials.unwrap_or(false);

        let mut module = Module::new();

        // LIF neurons
        module.push(lif_neuron_v1(
            20.0, // tau_m
            -70.0, // v_rest
            -70.0, // v_reset
            -50.0, // v_thresh
            2.0, // t_refrac
            10.0, // r_m
            1.0, // c_m
        ));

        // STDP plasticity
        module.push(stdp_rule_v1(
            0.01, // a_plus
            0.012, // a_minus
            20.0, // tau_plus
            20.0, // tau_minus
            0.0, // w_min
            1.0, // w_max
        ));

        // Input to hidden connections
        if inputs > 0 && hidden > 0 {
            module.push(layer_fully_connected_v1(
                0, // source_start
                inputs - 1, // source_end
                inputs, // target_start
                inputs + hidden - 1, // target_end
                1.0, // weight_mean
                1.0, // weight_std
            ));
        }

        // Hidden to output connections
        if hidden > 0 && outputs > 0 {
            module.push(layer_fully_connected_v1(
                inputs, // source_start
                inputs + hidden - 1, // source_end
                inputs + hidden, // target_start
                inputs + hidden + outputs - 1, // target_end
                1.0, // weight_mean
                1.0, // weight_std
            ));
        }

        // Poisson stimulus for inputs
        for i in 0..inputs {
            module.push(stimulus_poisson_v1(
                i, // neuron_id
                rate, // rate_hz
                10.0, // t_start_ms
                0.0, // weight
                duration, // t_end_ms
            ));
        }

        // Simulation runtime
        module.push(runtime_simulate_run_v1(
            dt, // dt_ms
            duration, // duration_ms
            record, // record_potentials
            seed, // seed
        ));

        Ok(module.to_text())
    }
}

/// Python wrapper for NIR execution results
#[pyclass(name = "NIRExecutionResult")]
pub struct PyNIRExecutionResult {
    pub spike_data: Vec<(u64, u32)>, // (time_ns, neuron_id)
    pub duration_ns: u64,
}

#[pymethods]
impl PyNIRExecutionResult {
    /// Get spike count
    fn spike_count(&self) -> usize {
        self.spike_data.len()
    }

    /// Get duration in nanoseconds
    fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// Get spikes as list of (time_ns, neuron_id) tuples
    fn spikes(&self) -> Vec<(u64, u32)> {
        self.spike_data.clone()
    }

    /// Export spikes in various formats
    #[pyo3(signature = (format="json"))]
    fn export_spikes(&self, format: Option<String>) -> PyResult<PyObject> {
        match format.as_deref().unwrap_or("json") {
            "json" => {
                Python::with_gil(|py| {
                    let list = PyList::empty(py);
                    for (time_ns, neuron_id) in &self.spike_data {
                        let spike_dict = PyDict::new(py);
                        spike_dict.set_item("neuron_id", neuron_id)?;
                        spike_dict.set_item("time_ns", time_ns)?;
                        spike_dict.set_item("time_ms", *time_ns as f64 / 1_000_000.0)?;
                        list.append(spike_dict)?;
                    }
                    Ok(list.to_object(py))
                })
            }
            "vevt" => {
                // Create VEVT binary format
                let events: Vec<VEVTEvent> = self.spike_data.iter().enumerate().map(|(i, (time_ns, neuron_id))| {
                    VEVTEvent {
                        timestamp: *time_ns,
                        event_type: 0, // Spike
                        source_id: *neuron_id,
                        target_id: u32::MAX,
                        payload_size: 0,
                        reserved: 0,
                    }
                }).collect();

                let bytes = encode_vevt(
                    StreamId::new(1),
                    StorageTime::from_nanos(0),
                    StorageTime::from_nanos(self.duration_ns),
                    &events
                ).map_err(|e| PyRuntimeError::new_err(format!("VEVT encoding failed: {:?}", e)))?;

                Python::with_gil(|py| {
                    Ok(pybytes::PyBytes::new(py, &bytes).to_object(py))
                })
            }
            "graphml" => {
                let mut graphml = String::new();
                graphml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
                graphml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");
                graphml.push_str("  <graph id=\"spikes\" edgedefault=\"directed\">\n");

                // Add nodes for neurons
                let mut neurons = std::collections::HashSet::new();
                for (_, neuron_id) in &self.spike_data {
                    neurons.insert(*neuron_id);
                }
                for neuron_id in neurons {
                    graphml.push_str(&format!("    <node id=\"neuron_{}\"/>\n", neuron_id));
                }

                // Add edges for spikes
                for (i, (time_ns, neuron_id)) in self.spike_data.iter().enumerate() {
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

                Python::with_gil(|py| {
                    Ok(graphml.to_object(py))
                })
            }
            _ => Err(PyValueError::new_err("Unsupported export format")),
        }
    }

    fn __repr__(&self) -> String {
        format!("NIRExecutionResult(spikes={}, duration={}ns)",
                self.spike_data.len(), self.duration_ns)
    }
}