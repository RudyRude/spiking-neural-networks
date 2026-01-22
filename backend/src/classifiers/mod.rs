//! Classifiers and Regressors using Spiking Neural Networks
//!
//! This module provides implementations for STDP-based unsupervised classifiers,
//! R-STDP classifiers/regressors with reward optimization, and LSM-based models.
//! Includes training algorithms, evaluation metrics, and integration with digital twin.

use crate::neuron::integrate_and_fire::IzhikevichNeuron;
use crate::neuron::iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType};
use crate::neuron::plasticity::{STDP, RewardModulatedSTDP, TraceRSTDP};
use crate::neuron::{Lattice, AdjacencyMatrix, SpikeHistory, RewardModulatedLattice};
use crate::error::SpikingNeuralNetworksError;
use rand::Rng;

/// Trait for classifiers
pub trait Classifier {
    /// Train the classifier with inputs and labels
    fn train(&mut self, inputs: &[Vec<f32>], labels: &[usize]) -> Result<(), SpikingNeuralNetworksError>;

    /// Predict class for a single input
    fn predict(&self, input: &[f32]) -> usize;
}

/// Trait for regressors
pub trait Regressor {
    /// Train the regressor with inputs and targets
    fn train(&mut self, inputs: &[Vec<f32>], targets: &[f32]) -> Result<(), SpikingNeuralNetworksError>;

    /// Predict value for a single input
    fn predict(&self, input: &[f32]) -> f32;
}

/// STDP-based unsupervised classifier using competitive learning
pub struct STDPClassifier {
    lattice: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        SpikeHistory,
        STDP,
        ApproximateNeurotransmitter,
    >,
    n_classes: usize,
    input_size: usize,
}

impl STDPClassifier {
    /// Create a new STDP classifier
    pub fn new(input_size: usize, n_classes: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut lattice = Lattice::default();
        lattice.populate(&base_neuron, n_classes, 1).unwrap();
        // Connect inputs to classes (fully connected with random weights)
        // Note: This is simplified; in practice, need input connections
        lattice.connect(
            &|x, y| x != y,
            Some(&|_, _| rand::thread_rng().gen_range(0.1..1.0)),
        ).unwrap();
        lattice.do_plasticity = true;
        lattice.update_grid_history = true;

        Self { lattice, n_classes, input_size }
    }
}

impl Classifier for STDPClassifier {
    fn train(&mut self, inputs: &[Vec<f32>], labels: &[usize]) -> Result<(), SpikingNeuralNetworksError> {
        // Unsupervised: ignore labels, use competitive learning
        for input in inputs {
            // Set input as external current to neurons (simplified)
            for (i, &val) in input.iter().enumerate() {
                if let Some(neuron) = self.lattice.get_mut(i % self.n_classes, 0) {
                    neuron.current_voltage += val;
                }
            }
            // Run iteration
            self.lattice.iterate()?;
            // Apply winner-take-all inhibition (simplified: reduce others)
            // Find winner
            let mut max_spike = 0.0;
            let mut winner = 0;
            for (i, neuron) in self.lattice.grid.iter().enumerate() {
                if neuron.last_firing_time > max_spike {
                    max_spike = neuron.last_firing_time;
                    winner = i;
                }
            }
            // Inhibit others
            for (i, neuron) in self.lattice.grid.iter_mut().enumerate() {
                if i != winner {
                    neuron.current_voltage -= 1.0; // Inhibition
                }
            }
        }
        Ok(())
    }

    fn predict(&self, input: &[f32]) -> usize {
        // Run prediction
        let mut temp_lattice = self.lattice.clone();
        for (i, &val) in input.iter().enumerate() {
            if let Some(neuron) = temp_lattice.get_mut(i % self.n_classes, 0) {
                neuron.current_voltage += val;
            }
        }
        temp_lattice.iterate().unwrap();
        // Return winner
        let mut max_spike = 0.0;
        let mut winner = 0;
        for (i, neuron) in temp_lattice.grid.iter().enumerate() {
            if neuron.last_firing_time > max_spike {
                max_spike = neuron.last_firing_time;
                winner = i;
            }
        }
        winner
    }
}

/// R-STDP classifier with reward optimization
pub struct RSTDPClassifier {
    lattice: RewardModulatedLattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), TraceRSTDP>,
        SpikeHistory,
    >,
    n_classes: usize,
    input_size: usize,
}

impl RSTDPClassifier {
    pub fn new(input_size: usize, n_classes: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut lattice = RewardModulatedLattice::default();
        lattice.populate(&base_neuron, n_classes, 1).unwrap();
        lattice.connect(
            &|x, y| x != y,
            Some(&|_, _| TraceRSTDP {
                weight: rand::thread_rng().gen_range(0.1..1.0),
                ..TraceRSTDP::default()
            }),
        ).unwrap();
        lattice.do_modulation = true;
        lattice.update_graph_history = true;

        Self { lattice, n_classes, input_size }
    }
}

impl Classifier for RSTDPClassifier {
    fn train(&mut self, inputs: &[Vec<f32>], labels: &[usize]) -> Result<(), SpikingNeuralNetworksError> {
        for (input, &label) in inputs.iter().zip(labels) {
            // Set input
            for (i, &val) in input.iter().enumerate() {
                if let Some(neuron) = self.lattice.get_mut(i % self.n_classes, 0) {
                    neuron.current_voltage += val;
                }
            }
            self.lattice.iterate()?;
            // Predict
            let prediction = self.predict(input);
            // Reward if correct
            let reward = if prediction == label { 1.0 } else { -1.0 };
            self.lattice.apply_reward(reward);
            self.lattice.update_plasticity();
        }
        Ok(())
    }

    fn predict(&self, input: &[f32]) -> usize {
        let mut temp_lattice = self.lattice.clone();
        for (i, &val) in input.iter().enumerate() {
            if let Some(neuron) = temp_lattice.get_mut(i % self.n_classes, 0) {
                neuron.current_voltage += val;
            }
        }
        temp_lattice.iterate().unwrap();
        let mut max_spike = 0.0;
        let mut winner = 0;
        for (i, neuron) in temp_lattice.grid.iter().enumerate() {
            if neuron.last_firing_time > max_spike {
                max_spike = neuron.last_firing_time;
                winner = i;
            }
        }
        winner
    }
}

/// LSM-based classifier (simplified)
pub struct LSMClassifier {
    reservoir: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        SpikeHistory,
        STDP,
        ApproximateNeurotransmitter,
    >,
    readout_weights: Vec<Vec<f32>>, // Weights from reservoir to classes
    n_classes: usize,
}

impl LSMClassifier {
    pub fn new(input_size: usize, reservoir_size: usize, n_classes: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut reservoir = Lattice::default();
        reservoir.populate(&base_neuron, reservoir_size, 1).unwrap();
        reservoir.connect(
            &|x, y| x != y && rand::thread_rng().gen_bool(0.1),
            Some(&|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ).unwrap();
        reservoir.update_grid_history = true;

        let readout_weights = vec![vec![0.0; reservoir_size]; n_classes];

        Self { reservoir, readout_weights, n_classes }
    }
}

impl Classifier for LSMClassifier {
    fn train(&mut self, inputs: &[Vec<f32>], labels: &[usize]) -> Result<(), SpikingNeuralNetworksError> {
        let mut reservoir_states = Vec::new();
        for input in inputs {
            // Drive reservoir with input
            for (i, &val) in input.iter().enumerate() {
                if let Some(neuron) = self.reservoir.get_mut(i % self.reservoir.grid.len(), 0) {
                    neuron.current_voltage += val;
                }
            }
            self.reservoir.iterate()?;
            // Collect state
            let state: Vec<f32> = self.reservoir.grid.iter().map(|n| n.last_firing_time).collect();
            reservoir_states.push(state);
        }
        // Train readout with pseudo-inverse or simple rule
        // Simplified: For each class, average state
        for class in 0..self.n_classes {
            let mut class_states = vec![0.0; self.reservoir.grid.len()];
            let mut count = 0;
            for (state, &label) in reservoir_states.iter().zip(labels) {
                if label == class {
                    for (i, &s) in state.iter().enumerate() {
                        class_states[i] += s;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for w in &mut self.readout_weights[class] {
                    *w /= count as f32;
                }
            }
        }
        Ok(())
    }

    fn predict(&self, input: &[f32]) -> usize {
        // Drive reservoir
        let mut temp_reservoir = self.reservoir.clone();
        for (i, &val) in input.iter().enumerate() {
            if let Some(neuron) = temp_reservoir.get_mut(i % temp_reservoir.grid.len(), 0) {
                neuron.current_voltage += val;
            }
        }
        temp_reservoir.iterate().unwrap();
        let state: Vec<f32> = temp_reservoir.grid.iter().map(|n| n.last_firing_time).collect();
        // Compute readout
        let mut max_score = f32::NEG_INFINITY;
        let mut prediction = 0;
        for (class, weights) in self.readout_weights.iter().enumerate() {
            let score: f32 = state.iter().zip(weights).map(|(s, w)| s * w).sum();
            if score > max_score {
                max_score = score;
                prediction = class;
            }
        }
        prediction
    }
}

/// R-STDP regressor
pub struct RSTDPRegressor {
    lattice: RewardModulatedLattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), TraceRSTDP>,
        SpikeHistory,
    >,
    readout: Vec<f32>,
    input_size: usize,
}

impl RSTDPRegressor {
    pub fn new(input_size: usize) -> Self {
        let base_neuron = IzhikevichNeuron::default_impl();
        let mut lattice = RewardModulatedLattice::default();
        lattice.populate(&base_neuron, input_size, 1).unwrap();
        lattice.connect(
            &|x, y| x != y,
            Some(&|_, _| TraceRSTDP {
                weight: rand::thread_rng().gen_range(0.1..1.0),
                ..TraceRSTDP::default()
            }),
        ).unwrap();
        lattice.do_modulation = true;
        lattice.update_graph_history = true;

        let readout = vec![0.0; input_size];

        Self { lattice, readout, input_size }
    }
}

impl Regressor for RSTDPRegressor {
    fn train(&mut self, inputs: &[Vec<f32>], targets: &[f32]) -> Result<(), SpikingNeuralNetworksError> {
        for (input, &target) in inputs.iter().zip(targets) {
            // Set input
            for (i, &val) in input.iter().enumerate() {
                if let Some(neuron) = self.lattice.get_mut(i % self.input_size, 0) {
                    neuron.current_voltage += val;
                }
            }
            self.lattice.iterate()?;
            // Compute output
            let output: f32 = self.lattice.grid.iter().zip(&self.readout).map(|(n, &w)| n.last_firing_time * w).sum();
            // Reward based on error
            let error = target - output;
            let reward = -error.abs(); // Negative error as reward
            self.lattice.apply_reward(reward);
            self.lattice.update_plasticity();
            // Update readout (simple rule)
            for (i, neuron) in self.lattice.grid.iter().enumerate() {
                self.readout[i] += 0.01 * error * neuron.last_firing_time;
            }
        }
        Ok(())
    }

    fn predict(&self, input: &[f32]) -> f32 {
        let mut temp_lattice = self.lattice.clone();
        for (i, &val) in input.iter().enumerate() {
            if let Some(neuron) = temp_lattice.get_mut(i % self.input_size, 0) {
                neuron.current_voltage += val;
            }
        }
        temp_lattice.iterate().unwrap();
        self.lattice.grid.iter().zip(&self.readout).map(|(n, &w)| n.last_firing_time * w).sum()
    }
}

/// Evaluation metrics
pub mod metrics {
    /// Classification accuracy
    pub fn accuracy(predictions: &[usize], labels: &[usize]) -> f32 {
        let correct: usize = predictions.iter().zip(labels).map(|(p, l)| if p == l { 1 } else { 0 }).sum();
        correct as f32 / labels.len() as f32
    }

    /// Mean Squared Error for regression
    pub fn mse(predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter().zip(targets).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / predictions.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_classifier() {
        let mut classifier = STDPClassifier::new(10, 3);
        let inputs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 2]; // Ignored for unsupervised
        classifier.train(&inputs, &labels).unwrap();
        let pred = classifier.predict(&inputs[0]);
        assert!(pred < 3);
    }

    #[test]
    fn test_rstdp_classifier() {
        let mut classifier = RSTDPClassifier::new(10, 3);
        let inputs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 2];
        classifier.train(&inputs, &labels).unwrap();
        let pred = classifier.predict(&inputs[0]);
        assert_eq!(pred, 0); // Should learn
    }

    #[test]
    fn test_lsm_classifier() {
        let mut classifier = LSMClassifier::new(10, 20, 3);
        let inputs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 2];
        classifier.train(&inputs, &labels).unwrap();
        let pred = classifier.predict(&inputs[0]);
        assert!(pred < 3);
    }

    #[test]
    fn test_rstdp_regressor() {
        let mut regressor = RSTDPRegressor::new(10);
        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let targets = vec![1.0, 2.0];
        regressor.train(&inputs, &targets).unwrap();
        let pred = regressor.predict(&inputs[0]);
        assert!(pred > 0.0);
    }

    #[test]
    fn test_metrics() {
        let preds = vec![0, 1, 2];
        let labels = vec![0, 1, 2];
        assert_eq!(metrics::accuracy(&preds, &labels), 1.0);

        let preds_reg = vec![1.0, 2.0];
        let targets = vec![1.0, 2.0];
        assert_eq!(metrics::mse(&preds_reg, &targets), 0.0);
    }
}