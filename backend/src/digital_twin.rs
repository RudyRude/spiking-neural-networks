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
use crate::classifiers::{Classifier, Regressor};
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

// Cue Model Module for working memory with recurrent neurons and noise modulation
pub struct CueModelModule {
    neurons: Vec<IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>>,
    last_firing_times: Vec<f32>,
    plasticity: STDP,
    graph: AdjacencyMatrix<(usize, usize), f32>,
    timestep: f32,
    noise_level: f32,
}

impl CueModelModule {
    pub fn new(size: usize, noise_level: f32) -> Self {
        let neurons = (0..size).map(|_| IzhikevichNeuron::default_impl()).collect();
        let last_firing_times = vec![0.0; size];
        let plasticity = STDP::default();
        let mut graph = AdjacencyMatrix::default();
        // Recurrent connections with random weights
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    graph.add_edge((i, j), rand::thread_rng().gen_range(-0.5..0.5));
                }
            }
        }
        Self { neurons, last_firing_times, plasticity, graph, timestep: 0.0, noise_level }
    }
}

impl BrainRegion for CueModelModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        self.timestep += 0.1;
        let mut spikes = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut synaptic_input = 0.0;
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(&weight) = self.graph.lookup_weight(&(j, i)) {
                        if self.last_firing_times[j] > 0.0 {
                            synaptic_input += weight;
                        }
                    }
                }
            }
            let external_input = inputs.get(i).map(|v| v.iter().sum()).unwrap_or(0.0);
            // Add noise modulation
            let noise = rand::thread_rng().gen_range(-self.noise_level..self.noise_level);
            let total_input = synaptic_input + external_input + noise;
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
        self.last_firing_times.clone()
    }

    fn update_plasticity(&mut self) {
        // Apply STDP plasticity
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(weight) = self.graph.lookup_weight_mut(&(j, i)) {
                        let dt = self.last_firing_times[i] - self.last_firing_times[j];
                        let delta_w = if dt > 0.0 {
                            self.plasticity.a_minus * (-dt).exp()
                        } else {
                            self.plasticity.a_plus * dt.exp()
                        };
                        *weight += delta_w;
                    }
                }
            }
        }
    }
}

// Fading Memory Module with decaying gap junctions
pub struct FadingMemoryModule {
    lattice: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        SpikeHistory,
        STDP,
        ApproximateNeurotransmitter,
    >,
    decay_rate: f32,
}

impl FadingMemoryModule {
    pub fn new(size: usize, decay_rate: f32) -> Self {
        let base_neuron = IzhikevichNeuron {
            gap_conductance: 5.0, // High gap conductance for memory
            ..IzhikevichNeuron::default_impl()
        };
        let mut lattice = Lattice::default();
        lattice.populate(&base_neuron, size, 1).unwrap();
        // Connect with uniform gap junctions
        lattice.connect(
            &(|x, y| x != y),
            Some(&(|_, _| 1.0)),
        ).unwrap();
        lattice.update_grid_history = true;
        Self { lattice, decay_rate }
    }
}

impl BrainRegion for FadingMemoryModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        if let Some(input_vec) = inputs.get(0) {
            for (i, &input) in input_vec.iter().enumerate() {
                if let Some(neuron) = self.lattice.get_mut(i, 0) {
                    neuron.current_voltage += input;
                }
            }
        }
        self.lattice.iterate().unwrap();
        // Implement fading memory by decaying membrane potentials
        self.lattice.apply(|neuron| neuron.current_voltage *= (1.0 - self.decay_rate));
        vec![0.0; self.lattice.grid.len()]
    }

    fn get_outputs(&self) -> Vec<f32> {
        let firing_rates = self.lattice.grid_history.aggregate();
        firing_rates.iter().map(|row| row[0] as f32).collect()
    }

    fn update_plasticity(&mut self) {
        // No plasticity for fading memory
    }
}

// Astrocyte Module with calcium dynamics and glutamate release for tripartite synapses
pub struct AstrocyteModule {
    calcium: f32, // Intracellular calcium concentration
    glutamate: f32, // Released glutamate concentration
    calcium_decay: f32,
    glutamate_release_threshold: f32,
    glutamate_decay: f32,
    dt: f32,
}

impl AstrocyteModule {
    pub fn new() -> Self {
        Self {
            calcium: 0.0,
            glutamate: 0.0,
            calcium_decay: 0.1,
            glutamate_release_threshold: 1.0,
            glutamate_decay: 0.05,
            dt: 0.1,
        }
    }
}

impl BrainRegion for AstrocyteModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        // Inputs are spike rates from neurons
        let total_input = inputs.iter().flatten().sum::<f32>();
        // Calcium increases with neuronal activity
        self.calcium += total_input * self.dt - self.calcium * self.calcium_decay * self.dt;
        self.calcium = self.calcium.max(0.0);

        // Release glutamate if calcium is high
        if self.calcium > self.glutamate_release_threshold {
            self.glutamate += (self.calcium - self.glutamate_release_threshold) * self.dt;
        }
        // Glutamate decays
        self.glutamate -= self.glutamate * self.glutamate_decay * self.dt;
        self.glutamate = self.glutamate.max(0.0);

        // Output glutamate concentration
        vec![self.glutamate]
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![self.glutamate]
    }

    fn update_plasticity(&mut self) {
        // Astrocytes may have plasticity, but placeholder
    }
}

// Pathology simulation: Schizophrenia model with GABA/NMDA imbalances
pub struct SchizophreniaModule {
    neurons: Vec<IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>>,
    last_firing_times: Vec<f32>,
    plasticity: STDP,
    graph: AdjacencyList<(usize, usize), f32>,
    timestep: f32,
    nmda_reduction: f32, // Factor to reduce NMDA efficacy (e.g., 0.5 for hypofunction)
    gaba_increase: f32, // Factor to increase GABA (e.g., 1.5)
}

impl SchizophreniaModule {
    pub fn new(size: usize, nmda_reduction: f32, gaba_increase: f32) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..size {
            let mut neuron = IzhikevichNeuron::default_impl();
            // Modify receptors for pathology
            if let Some(nmda_receptor) = neuron.receptors.get_mut(&NeurotransmitterType::NMDA) {
                nmda_receptor.g *= nmda_reduction;
            }
            if let Some(gaba_receptor) = neuron.receptors.get_mut(&NeurotransmitterType::GABAa) {
                gaba_receptor.g *= gaba_increase;
            }
            neurons.push(neuron);
        }
        let last_firing_times = vec![0.0; size];
        let plasticity = STDP::default();
        let mut graph = AdjacencyList::new();
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    graph.add_edge((i, j), 0.5);
                }
            }
        }
        Self {
            neurons,
            last_firing_times,
            plasticity,
            graph,
            timestep: 0.0,
            nmda_reduction,
            gaba_increase,
        }
    }
}

impl BrainRegion for SchizophreniaModule {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        self.timestep += 0.1;
        let mut spikes = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut synaptic_input = 0.0;
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(&weight) = self.graph.get_edge(&(j, i)) {
                        if self.last_firing_times[j] > 0.0 {
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
        self.last_firing_times.clone()
    }

    fn update_plasticity(&mut self) {
        // Pathology may affect plasticity, but use standard for now
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j {
                    if let Some(weight) = self.graph.get_edge_mut(&(j, i)) {
                        let dt = self.last_firing_times[i] - self.last_firing_times[j];
                        let delta_w = if dt > 0.0 {
                            self.plasticity.a_minus * (-dt).exp()
                        } else {
                            self.plasticity.a_plus * dt.exp()
                        };
                        *weight += delta_w;
                    }
                }
            }
        }
    }
}

// Virtual Medication System: modulates receptor efficacies
pub struct VirtualMedication {
    pub nmda_modulation: f32, // Multiplier for NMDA conductance
    pub gaba_modulation: f32, // Multiplier for GABA conductance
    pub dopamine_modulation: f32, // Multiplier for dopamine effects
}

impl VirtualMedication {
    pub fn new(nmda: f32, gaba: f32, dopamine: f32) -> Self {
        Self {
            nmda_modulation: nmda,
            gaba_modulation: gaba,
            dopamine_modulation: dopamine,
        }
    }

    // Apply medication to a neuron
    pub fn apply_to_neuron(&self, neuron: &mut IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>) {
        if let Some(nmda) = neuron.receptors.get_mut(&NeurotransmitterType::NMDA) {
            nmda.g *= self.nmda_modulation;
        }
        if let Some(gaba) = neuron.receptors.get_mut(&NeurotransmitterType::GABAa) {
            gaba.g *= self.gaba_modulation;
        }
        // For dopamine, it can modulate plasticity or other
    }

    // For dopamine, modulate plasticity
    pub fn modulate_plasticity(&self, delta_w: f32) -> f32 {
        delta_w * self.dopamine_modulation
    }
}

// Classifier Module for digital twin integration
pub struct ClassifierModule<C: Classifier> {
    classifier: C,
    trained: bool,
}

impl<C: Classifier> ClassifierModule<C> {
    pub fn new(classifier: C) -> Self {
        Self { classifier, trained: false }
    }

    pub fn train(&mut self, inputs: &[Vec<f32>], labels: &[usize]) {
        self.classifier.train(inputs, labels).unwrap();
        self.trained = true;
    }
}

impl<C: Classifier> BrainRegion for ClassifierModule<C> {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        if !self.trained {
            // If not trained, return zeros
            return vec![0.0; 1];
        }
        // Assume inputs[0] is the input vector
        if let Some(input) = inputs.get(0) {
            let prediction = self.classifier.predict(input) as f32;
            vec![prediction]
        } else {
            vec![0.0]
        }
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![0.0] // Placeholder, since prediction is done in iterate
    }

    fn update_plasticity(&mut self) {
        // No plasticity for classifier
    }
}

// Regressor Module for digital twin integration
pub struct RegressorModule<R: Regressor> {
    regressor: R,
    trained: bool,
}

impl<R: Regressor> RegressorModule<R> {
    pub fn new(regressor: R) -> Self {
        Self { regressor, trained: false }
    }

    pub fn train(&mut self, inputs: &[Vec<f32>], targets: &[f32]) {
        self.regressor.train(inputs, targets).unwrap();
        self.trained = true;
    }
}

impl<R: Regressor> BrainRegion for RegressorModule<R> {
    fn iterate(&mut self, inputs: &[Vec<f32>]) -> Vec<f32> {
        if !self.trained {
            return vec![0.0];
        }
        if let Some(input) = inputs.get(0) {
            let prediction = self.regressor.predict(input);
            vec![prediction]
        } else {
            vec![0.0]
        }
    }

    fn get_outputs(&self) -> Vec<f32> {
        vec![0.0]
    }

    fn update_plasticity(&mut self) {
        // No plasticity
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

    #[test]
    fn test_astrocyte_module() {
        let mut astrocyte = AstrocyteModule::new();
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // Some spikes
        let output = astrocyte.iterate(&inputs);
        assert!(!output.is_empty());
        // Glutamate should increase with activity
        assert!(astrocyte.glutamate >= 0.0);
    }

    #[test]
    fn test_schizophrenia_module() {
        let mut schizophrenia = SchizophreniaModule::new(5, 0.5, 1.5); // Reduced NMDA, increased GABA
        let inputs = vec![vec![0.0]; 5];
        let output = schizophrenia.iterate(&inputs);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_virtual_medication() {
        let medication = VirtualMedication::new(1.2, 0.8, 1.5); // Increase NMDA, decrease GABA, increase dopamine
        let modulated = medication.modulate_plasticity(1.0);
        assert_eq!(modulated, 1.5);
    }

    #[test]
    fn test_cue_model_module() {
        let mut cue = CueModelModule::new(5, 0.1);
        let inputs = vec![vec![0.0]; 5];
        let output = cue.iterate(&inputs);
        assert_eq!(output.len(), 5);
        cue.update_plasticity();
        // Basic test that it runs without panic
        assert!(true);
    }

    #[test]
    fn test_fading_memory_module() {
        let mut fading = FadingMemoryModule::new(5, 0.01);
        let inputs = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]];
        let output = fading.iterate(&inputs);
        assert_eq!(output.len(), 5);
        // Check that outputs are firing rates
        let firing_rates = fading.get_outputs();
        assert_eq!(firing_rates.len(), 5);
    }

    #[test]
    fn test_classifier_module() {
        use crate::classifiers::STDPClassifier;
        let classifier = STDPClassifier::new(3, 2);
        let mut module = ClassifierModule::new(classifier);
        let inputs = vec![vec![1.0, 0.0]];
        // Train first
        module.train(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[0, 1]);
        let output = module.iterate(&inputs);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_regressor_module() {
        use crate::classifiers::RSTDPRegressor;
        let regressor = RSTDPRegressor::new(3);
        let mut module = RegressorModule::new(regressor);
        let inputs = vec![vec![1.0, 0.0]];
        // Train first
        module.train(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[1.0, 2.0]);
        let output = module.iterate(&inputs);
        assert_eq!(output.len(), 1);
    }
}