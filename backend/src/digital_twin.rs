//! Digital Twin of Brain Architecture
//!
//! This module provides a modular framework for simulating brain regions
//! integrated into a cohesive digital twin. It combines various neuron models,
//! plasticity rules, attractors, and liquid state machines from the documentation.

use crate::neuron::iterate_and_spike::{IterateAndSpike, ApproximateNeurotransmitter, ApproximateReceptor, LastFiringTime};
use crate::graph::{Graph, AdjacencyList};
use crate::neuron::plasticity::STDP;
use crate::neuron::integrate_and_fire::IzhikevichNeuron;
use std::collections::HashMap;

/// Trait for a brain region module.
/// Each region can iterate its internal state, receive inputs, and produce outputs.
pub trait BrainRegion {
    /// Iterate the region's dynamics for one time step.
    /// inputs: External inputs as a vector of spike trains or currents.
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32>;

    /// Get the region's output (e.g., firing rates or spike trains).
    fn get_outputs(&self) -> Vec<f32>;

    /// Update internal plasticity based on activity.
    fn update_plasticity(&mut self);
}

/// Digital Twin orchestrator.
/// Manages multiple brain regions connected via a graph.
pub struct DigitalTwin {
    regions: HashMap<String, Box<dyn BrainRegion>>,
    connectivity: Graph<f32>, // Weights between regions
}

impl DigitalTwin {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
            connectivity: Graph::new(),
        }
    }

    /// Add a region to the twin.
    pub fn add_region(&mut self, name: String, region: Box<dyn BrainRegion>) {
        self.regions.insert(name, region);
    }

    /// Connect regions with a weight.
    pub fn connect_regions(&mut self, from: &str, to: &str, weight: f32) {
        // Assume graph has node indices; for simplicity, use string keys
        // In practice, map strings to indices.
    }

    /// Run one time step of the entire twin.
    pub fn iterate(&mut self) {
        // Collect outputs from all regions.
        let mut region_outputs: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, region) in &self.regions {
            let inputs = vec![]; // For now, no inter-region inputs
            let outputs = region.iterate(&inputs);
            region_outputs.insert(name.clone(), outputs);
        }

        // Update plasticity for each region.
        for region in self.regions.values_mut() {
            region.update_plasticity();
        }
    }
}

// Example: Cortical Module with Izhikevich neurons and STDP.
pub struct CorticalModule {
    neurons: Vec<IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>>,
    last_firing_times: Vec<f32>,
    plasticity: STDP,
    graph: AdjacencyList<(usize, usize), f32>,
    timestep: f32,
}

impl CorticalModule {
    pub fn new(size: usize) -> Self {
        let neurons = (0..size).map(|_| IzhikevichNeuron::default_impl()).collect();
        let last_firing_times = vec![0.0; size];
        let plasticity = STDP::default();
        let mut graph = AdjacencyList::new();
        // Initialize full connectivity for simplicity
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    graph.add_edge((i, j), 0.5); // Random weight
                }
            }
        }
        Self { neurons, last_firing_times, plasticity, graph, timestep: 0.0 }
    }
}

impl BrainRegion for CorticalModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        self.timestep += 0.1; // Assume dt=0.1
        let mut spikes = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate synaptic input from graph (simplified: sum weighted spikes from all others)
            let mut synaptic_input = 0.0;
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(&weight) = self.graph.get_edge(&(j, i)) {
                        if self.last_firing_times[j] > 0.0 { // If recently spiked
                            synaptic_input += weight;
                        }
                    }
                }
            }
            let external_input = inputs.get(i).map(|v| v.iter().sum()).unwrap_or(0.0);
            let total_input = synaptic_input + external_input;
            let spiked = neuron.iterate_and_spike(total_input);
            if spiked {
                self.last_firing_times[i] = self.timestep;
                spikes.push(1.0);
            } else {
                spikes.push(0.0);
            }
        }
        spikes
    }

    fn get_outputs(&self) -> Vec<f32> {
        self.last_firing_times.clone() // Return last firing times as output
    }

    fn update_plasticity(&mut self) {
        // Apply STDP: for each pair, if postsynaptic spiked, update based on timing
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(weight) = self.graph.get_edge_mut(&(j, i)) {
                        let dt = self.last_firing_times[i] - self.last_firing_times[j];
                        if dt > 0.0 { // Post after pre
                            *weight += self.plasticity.a_minus * (-dt).exp();
                        } else { // Pre after post
                            *weight += self.plasticity.a_plus * dt.exp();
                        }
                    }
                }
            }
        }
    }
}

// Placeholder for Hippocampal Module (Ring Attractor).
pub struct HippocampalModule {
    // Implement ring attractor with Gaussian weights.
}

impl BrainRegion for HippocampalModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        vec![] // Placeholder
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![] // Placeholder
    }

    fn update_plasticity(&mut self) {}
}

// Placeholder for LSM Module.
pub struct LsmModule {
    // Reservoir with readout.
}

impl BrainRegion for LsmModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        vec![] // Placeholder
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![] // Placeholder
    }

    fn update_plasticity(&mut self) {}
}