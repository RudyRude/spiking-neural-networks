//! Digital Twin of Brain Architecture
//!
//! This module provides a modular framework for simulating brain regions
//! integrated into a cohesive digital twin. It combines various neuron models,
//! plasticity rules, attractors, and liquid state machines from the documentation.

use crate::neuron::iterate_and_spike::{IterateAndSpike, ApproximateNeurotransmitter, ApproximateReceptor, LastFiringTime};
use crate::graph::{Graph, AdjacencyList, AdjacencyMatrix};
use crate::neuron::plasticity::STDP;
use crate::neuron::integrate_and_fire::IzhikevichNeuron;
use crate::neuron::{Lattice, SpikeHistory, RunLattice};
use rand::Rng;
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
    dopamine: f32, // Neuromodulator
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
        Self { neurons, last_firing_times, plasticity, graph, timestep: 0.0, dopamine: 1.0 }
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
        // Apply R-STDP modulated by dopamine
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(weight) = self.graph.get_edge_mut(&(j, i)) {
                        let dt = self.last_firing_times[i] - self.last_firing_times[j];
                        let delta_w = if dt > 0.0 { // Post after pre
                            self.plasticity.a_minus * (-dt).exp()
                        } else { // Pre after post
                            self.plasticity.a_plus * dt.exp()
                        };
                        *weight += delta_w * self.dopamine; // Modulate by dopamine
                    }
                }
            }
        }
    }
}

// Hippocampal Module with Ring Attractor.
pub struct HippocampalModule {
    lattice: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        SpikeHistory,
        STDP,
        ApproximateNeurotransmitter,
    >,
    n_neurons: usize,
    preferred_direction: usize,
}

impl HippocampalModule {
    pub fn new(n_neurons: usize, preferred_direction: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut lattice = Lattice::default();
        lattice.populate(&base_neuron, n_neurons, 1).unwrap();
        // Set up ring connections
        let ring_distance = |x: isize, y: isize| -> f32 {
            (x - y).abs().min(n_neurons as isize - (x - y).abs()) as f32
        };
        lattice.connect(
            &(|x, y| x != y),
            Some(&(|x, y|
                (-2. * ring_distance(x.0 as isize, y.0 as isize).powf(2.) /
                (n_neurons as f32 * 10.)).exp() - 0.3)
            ),
        ).unwrap();
        // Random initial voltages
        lattice.apply(|neuron|
            neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_init..=neuron.v_th)
        );
        lattice.update_grid_history = true;
        Self { lattice, n_neurons, preferred_direction }
    }
}

impl BrainRegion for HippocampalModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        // Apply input to preferred direction
        if let Some(input_vec) = inputs.get(0) {
            if let Some(input) = input_vec.get(0) {
                // Input to preferred direction neuron
                if let Some(neuron) = self.lattice.get_mut(self.preferred_direction, 0) {
                    neuron.current_voltage += input;
                }
            }
        }
        // Run one step
        self.lattice.iterate().unwrap();
        // Return firing rates (simplified)
        vec![0.0; self.n_neurons] // Placeholder for actual rates
    }

    fn get_outputs(&self) -> Vec<f32> {
        // Aggregate firing rates
        let firing_rates = self.lattice.grid_history.aggregate();
        firing_rates.iter().map(|row| row[0] as f32).collect()
    }

    fn update_plasticity(&mut self) {
        // No plasticity for attractor
    }
}

// LSM Module with reservoir and readout.
pub struct LsmModule {
    reservoir: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        SpikeHistory,
        STDP,
        ApproximateNeurotransmitter,
    >,
    readout: Vec<f32>, // Simple readout weights
    output: f32,
}

impl LsmModule {
    pub fn new(reservoir_size: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut reservoir = Lattice::default();
        reservoir.populate(&base_neuron, reservoir_size, 1).unwrap();
        // Random recurrent connections
        reservoir.connect(
            &(|x, y| x != y && rand::thread_rng().gen_bool(0.1)), // 10% connectivity
            Some(&(|_, _| rand::thread_rng().gen_range(-1.0..1.0))),
        ).unwrap();
        reservoir.update_grid_history = true;
        let readout = vec![0.0; reservoir_size]; // Initialize to zero
        Self { reservoir, readout, output: 0.0 }
    }
}

impl BrainRegion for LsmModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        // Apply inputs to reservoir
        if let Some(input_vec) = inputs.get(0) {
            for (i, &input) in input_vec.iter().enumerate() {
                if let Some(neuron) = self.reservoir.get_mut(i, 0) {
                    neuron.current_voltage += input;
                }
            }
        }
        // Iterate reservoir
        self.reservoir.iterate().unwrap();
        // Compute readout
        let firing_rates = self.reservoir.grid_history.aggregate();
        self.output = firing_rates.iter().zip(&self.readout).map(|(rate, &weight)| rate[0] as f32 * weight).sum();
        vec![self.output]
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![self.output]
    }

    fn update_plasticity(&mut self) {
        // Simple R-STDP like update (placeholder)
        // For simplicity, no update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digital_twin() {
        let mut twin = DigitalTwin::new();
        let cortical = CorticalModule::new(5);
        let hippocampal = HippocampalModule::new(10, 5);
        twin.add_region("cortical".to_string(), Box::new(cortical));
        twin.add_region("hippocampal".to_string(), Box::new(hippocampal));
        for _ in 0..10 {
            twin.iterate();
        }
        // Basic test that it runs without panic
        assert!(true);
    }
}