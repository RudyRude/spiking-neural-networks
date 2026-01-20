//! Neuron models and dynamics for spiking neural networks
//!
//! This module provides various biologically-inspired neuron models including
//! Leaky Integrate-and-Fire (LIF), Hodgkin-Huxley, and Izhikevich neurons.
//! Each model implements the `Neuron` trait for consistent behavior across the framework.

use crate::spike::Spike;
use crate::time::TimeStep;

// Re-export the canonical NeuronId from spike module to ensure type consistency
pub use crate::spike::NeuronId;

/// Enumeration of available neuron types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NeuronType {
    /// Leaky Integrate-and-Fire neuron
    LIF,
    /// Adaptive Exponential Integrate-and-Fire neuron
    AdEx,
    /// Izhikevich neuron model
    Izhikevich,
}

impl Default for NeuronType {
    fn default() -> Self {
        Self::LIF
    }
}

/// Collection of neurons for efficient management
#[derive(Debug, Clone)]
pub struct NeuronPool<T: Neuron> {
    neurons: Vec<T>,
    active_indices: Vec<usize>,
}

impl<T: Neuron> NeuronPool<T> {
    /// Create a new empty neuron pool
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            active_indices: Vec::new(),
        }
    }

    /// Create a neuron pool with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            neurons: Vec::with_capacity(capacity),
            active_indices: Vec::with_capacity(capacity),
        }
    }

    /// Add a neuron to the pool
    pub fn add_neuron(&mut self, neuron: T) -> usize {
        let index = self.neurons.len();
        self.neurons.push(neuron);
        self.active_indices.push(index);
        index
    }

    /// Get a reference to a neuron by index
    pub fn get_neuron(&self, index: usize) -> Option<&T> {
        self.neurons.get(index)
    }

    /// Get a mutable reference to a neuron by index
    pub fn get_neuron_mut(&mut self, index: usize) -> Option<&mut T> {
        self.neurons.get_mut(index)
    }

    /// Get the number of neurons in the pool
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Get iterator over all neurons
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.neurons.iter()
    }

    /// Get mutable iterator over all neurons
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.neurons.iter_mut()
    }

    /// Update all neurons and collect generated spikes
    pub fn update_all(&mut self, dt: TimeStep) -> Vec<(usize, Spike)> {
        let mut spikes = Vec::new();
        for (index, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(spike) = neuron.update(dt) {
                spikes.push((index, spike));
            }
        }
        spikes
    }

    /// Reset all neurons to their default state
    pub fn reset_all(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset();
        }
    }
}

impl<T: Neuron> Default for NeuronPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Core trait for all neuron models
pub trait Neuron: Send + Sync + Clone {
    /// Integrate input current over time step
    fn integrate(&mut self, input_current: f64, dt: TimeStep);

    /// Update neuron state and check for spike generation
    fn update(&mut self, dt: TimeStep) -> Option<Spike>;

    /// Get current membrane potential
    fn membrane_potential(&self) -> f64;

    /// Set membrane potential (for testing/initialization)
    fn set_membrane_potential(&mut self, voltage: f64);

    /// Get spike threshold
    fn threshold(&self) -> f64;

    /// Reset neuron to post-spike state
    fn reset(&mut self);

    /// Get neuron's unique identifier
    fn id(&self) -> NeuronId;

    /// Set neuron's identifier
    fn set_id(&mut self, id: NeuronId);
}

/// Leaky Integrate-and-Fire neuron model
/// The LIF model is the simplest spiking neuron model, where the membrane potential
/// Configuration for LIF neuron parameters
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LIFConfig {
    /// Membrane time constant in milliseconds
    pub tau_membrane: f64,
    /// Membrane resistance in MegaOhms
    pub resistance: f64,
    /// Membrane capacitance in nanoFarads
    pub capacitance: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Reset potential after spike in millivolts
    pub reset_potential: f64,
    /// Resting potential in millivolts
    pub resting_potential: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
}

impl Default for LIFConfig {
    fn default() -> Self {
        Self {
            tau_membrane: 20.0,      // 20ms time constant
            resistance: 10.0,        // 10 MΩ resistance
            capacitance: 2.0,        // 2 nF capacitance
            threshold: -55.0,        // -55mV threshold
            reset_potential: -70.0,  // -70mV reset
            resting_potential: -65.0, // -65mV resting
            refractory_period: 2.0,  // 2ms refractory period
        }
    }
}

/// integrates input current with exponential decay (leak).
#[derive(Debug, Clone, PartialEq)]
pub struct LIFNeuron {
    id: NeuronId,
    state: NeuronState,

    // Parameters
    /// Membrane time constant in milliseconds
    pub tau_membrane: f64,
    /// Membrane resistance in MegaOhms
    pub resistance: f64,
    /// Membrane capacitance in nanoFarads
    pub capacitance: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Reset potential after spike in millivolts
    pub reset_potential: f64,
    /// Resting potential in millivolts
    pub resting_potential: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
}

/// Current state of a neuron including membrane potential and internal variables
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuronState {
    /// Current membrane potential in millivolts
    pub membrane_potential: f64,
    /// Remaining refractory period in timesteps
    pub refractory_timer: TimeStep,
    /// Timestamp of the last spike generated
    pub last_spike_time: Option<TimeStep>,
}

impl NeuronState {
    /// Create new default neuron state
    pub fn new() -> Self {
        Self {
            membrane_potential: -65.0, // Typical resting potential
            refractory_timer: 0,
            last_spike_time: None,
        }
    }

    /// Get current membrane potential
    pub fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    /// Check if neuron is in refractory period
    pub fn is_refractory(&self) -> bool {
        self.refractory_timer > 0
    }
}

impl Default for NeuronState {
    fn default() -> Self {
        Self::new()
    }
}

impl LIFNeuron {
    /// Create new LIF neuron with default parameters
    pub fn new(id: NeuronId) -> Self {
        Self::with_config(id, LIFConfig::default())
    }

    /// Create new LIF neuron with specified configuration
    pub fn with_config(id: NeuronId, config: LIFConfig) -> Self {
        Self {
            id,
            state: NeuronState::new(),
            tau_membrane: config.tau_membrane,
            resistance: config.resistance,
            capacitance: config.capacitance,
            threshold: config.threshold,
            reset_potential: config.reset_potential,
            resting_potential: config.resting_potential,
            refractory_period: config.refractory_period,
        }
    }

    /// Create LIF neuron with custom parameters
    pub fn with_params(
        id: NeuronId,
        tau_membrane: f64,
        threshold: f64,
        reset_potential: f64,
        refractory_period: f64,
    ) -> Self {
        let mut neuron = Self::new(id);
        neuron.tau_membrane = tau_membrane;
        neuron.threshold = threshold;
        neuron.reset_potential = reset_potential;
        neuron.refractory_period = refractory_period;
        neuron
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(NeuronId(0))
    }
}

impl Neuron for LIFNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.state.is_refractory() {
            // Update refractory timer
            self.state.refractory_timer = self.state.refractory_timer.saturating_sub(dt);
            return;
        }

        let dt_ms = dt as f64 / 1000.0; // Convert from TimeStep (u64) to milliseconds

        // Membrane equation: dV/dt = (V_rest - V)/tau + I*R/tau
        let leak_current = (self.resting_potential - self.state.membrane_potential) / self.tau_membrane;
        let input_term = input_current * self.resistance / self.tau_membrane;

        let dv_dt = leak_current + input_term;
        self.state.membrane_potential += dv_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= self.threshold {
            self.reset();
            self.state.last_spike_time = Some(0); // Would need current time
            self.state.refractory_timer = (self.refractory_period * 1000.0) as TimeStep;

            // Create spike with proper type conversion and error handling
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0), // Convert TimeStep to Time
                1.0 // Default spike amplitude
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.state.membrane_potential = self.reset_potential;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Adaptive Exponential Integrate-and-Fire neuron model
/// The AdEx model includes an exponential term and adaptation current,
/// providing more realistic spike generation and frequency adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct AdExNeuron {
    id: NeuronId,
    state: NeuronState,
    adaptation_current: f64,

    // Parameters
    /// Membrane time constant in milliseconds
    pub tau_membrane: f64,
    /// Adaptation time constant in milliseconds
    pub tau_adaptation: f64,
    /// Slope factor in millivolts for exponential threshold
    pub delta_t: f64,
    /// Leak conductance in nanoSiemens
    pub conductance: f64,
    /// Membrane capacitance in picoFarads
    pub capacitance: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Reset potential in millivolts
    pub reset_potential: f64,
    /// Resting potential in millivolts
    pub resting_potential: f64,
    /// Spike-triggered adaptation increment in picoAmperes
    pub adaptation_increment: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
}

impl AdExNeuron {
    /// Create new AdEx neuron with default parameters
    pub fn new(id: NeuronId) -> Self {
        Self {
            id,
            state: NeuronState::new(),
            adaptation_current: 0.0,
            tau_membrane: 9.3,       // 9.3ms membrane time constant
            tau_adaptation: 144.0,   // 144ms adaptation time constant
            delta_t: 2.0,            // 2mV slope factor
            conductance: 30.0,       // 30nS leak conductance
            capacitance: 281.0,      // 281pF capacitance
            threshold: -50.4,        // -50.4mV threshold
            reset_potential: -70.6,  // -70.6mV reset
            resting_potential: -70.6, // -70.6mV resting
            adaptation_increment: 4.0, // 4pA adaptation increment
            refractory_period: 2.0,   // 2ms refractory
        }
    }

    /// Get current adaptation current value
    pub fn adaptation_current(&self) -> f64 {
        self.adaptation_current
    }
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new(NeuronId(0))
    }
}

impl Neuron for AdExNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.state.is_refractory() {
            self.state.refractory_timer = self.state.refractory_timer.saturating_sub(dt);
            return;
        }

        let dt_ms = dt as f64 / 1000.0; // Convert from TimeStep (u64) to milliseconds
        let v = self.state.membrane_potential;

        // Exponential term for spike generation
        let exp_term = if v - self.threshold < 10.0 { // Avoid overflow
            self.delta_t * ((v - self.threshold) / self.delta_t).exp()
        } else {
            self.delta_t * (10.0f64).exp() // Large value to trigger spike
        };

        // Membrane equation with exponential term
        let leak_current = self.conductance * (self.resting_potential - v);
        let adaptation_term = -self.adaptation_current;
        let exponential_current = self.conductance * exp_term;

        let dv_dt = (leak_current + adaptation_term + exponential_current + input_current) / self.capacitance;

        // Update membrane potential
        self.state.membrane_potential += dv_dt * dt_ms;

        // Update adaptation current
        let da_dt = -self.adaptation_current / self.tau_adaptation;
        self.adaptation_current += da_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= self.threshold + 10.0 { // Spike condition
            self.reset();
            self.adaptation_current += self.adaptation_increment;
            self.state.refractory_timer = (self.refractory_period * 1000.0) as TimeStep;

            // Create spike with proper type conversion and error handling
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0 // Default spike amplitude
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.state.membrane_potential = self.reset_potential;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Izhikevich neuron model
/// A computationally efficient model that can reproduce various firing patterns
/// depending on parameter values.
#[derive(Debug, Clone, PartialEq)]
pub struct IzhikevichNeuron {
    id: NeuronId,
    state: NeuronState,
    recovery_variable: f64,

    // Parameters
    /// Recovery time constant in 1/ms
    pub a: f64,
    /// Recovery sensitivity in pA/mV
    pub b: f64,
    /// Reset potential in millivolts
    pub c: f64,
    /// Recovery increment in picoAmperes
    pub d: f64,
}

impl IzhikevichNeuron {
    /// Create new Izhikevich neuron with specified parameters
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            id: NeuronId(0),
            state: NeuronState::new(),
            recovery_variable: -14.0, // Typical initial value
            a,
            b,
            c,
            d,
        }
    }

    /// Create a regular spiking neuron
    pub fn regular_spiking(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.2, -65.0, 8.0);
        neuron.id = id;
        neuron
    }

    /// Create an intrinsically bursting neuron
    pub fn intrinsically_bursting(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.25, -65.0, 2.0);
        neuron.id = id;
        neuron
    }

    /// Create a chattering neuron
    pub fn chattering(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.02, 0.2, -50.0, 2.0);
        neuron.id = id;
        neuron
    }

    /// Create a fast spiking neuron
    pub fn fast_spiking(id: NeuronId) -> Self {
        let mut neuron = Self::new(0.1, 0.2, -65.0, 2.0);
        neuron.id = id;
        neuron
    }

    /// Get current recovery variable value
    pub fn recovery_variable(&self) -> f64 {
        self.recovery_variable
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::regular_spiking(NeuronId(0))
    }
}

impl Neuron for IzhikevichNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        let dt_ms = dt as f64 / 1000.0; // Convert from TimeStep (u64) to milliseconds
        let v = self.state.membrane_potential;
        let u = self.recovery_variable;

        // Izhikevich equations
        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        // du/dt = a*(b*v - u)

        let dv_dt = 0.04 * v * v + 5.0 * v + 140.0 - u + input_current;
        let du_dt = self.a * (self.b * v - u);

        self.state.membrane_potential += dv_dt * dt_ms;
        self.recovery_variable += du_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.state.membrane_potential >= 30.0 { // Fixed threshold for Izhikevich
            self.state.membrane_potential = self.c;
            self.recovery_variable += self.d;

            // Create spike with proper type conversion and error handling
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0 // Default spike amplitude
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.state.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.state.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        30.0 // Fixed threshold for Izhikevich model
    }

    fn reset(&mut self) {
        self.state.membrane_potential = self.c;
        self.recovery_variable += self.d;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Detailed Leaky Integrate-and-Fire neuron model
/// More biologically realistic with proper membrane equation
#[derive(Debug, Clone, PartialEq)]
pub struct DetailedLIFNeuron {
    /// Neuron identifier
    pub id: NeuronId,
    /// Current membrane potential in millivolts
    pub membrane_potential: f64,
    /// Resting potential in millivolts
    pub resting_potential: f64,
    /// Reset potential after spike in millivolts
    pub reset_potential: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Membrane time constant in milliseconds
    pub tau_membrane: f64,
    /// Membrane resistance in MegaOhms
    pub resistance: f64,
    /// Membrane capacitance in nanoFarads
    pub capacitance: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
    /// Remaining refractory time
    pub refractory_timer: TimeStep,
}

impl DetailedLIFNeuron {
    /// Create a DetailedLIFNeuron with default parameters from spiking-networks
    pub fn from_spiking_networks_defaults(id: NeuronId) -> Self {
        Self {
            id,
            membrane_potential: -65.0, // mV
            resting_potential: -65.0,  // mV
            reset_potential: -75.0,    // mV
            threshold: -55.0,          // mV
            tau_membrane: 20.0,        // ms
            resistance: 10.0,          // MΩ
            capacitance: 2.0,          // nF
            refractory_period: 2.0,    // ms
            refractory_timer: 0,
        }
    }
}

impl Neuron for DetailedLIFNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.refractory_timer > 0 {
            self.refractory_timer = self.refractory_timer.saturating_sub(dt);
            return;
        }

        let dt_ms = dt as f64 / 1000.0; // Convert to milliseconds

        // Membrane equation: dV/dt = (V_rest - V)/tau + I*R/tau
        let leak_current = (self.resting_potential - self.membrane_potential) / self.tau_membrane;
        let input_term = input_current * self.resistance / self.tau_membrane;

        let dv_dt = leak_current + input_term;
        self.membrane_potential += dv_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.membrane_potential >= self.threshold {
            self.reset();
            self.refractory_timer = (self.refractory_period * 1000.0) as TimeStep;

            // Create spike
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0), // Would use current time in real implementation
                1.0
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.membrane_potential = self.reset_potential;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Detailed Hodgkin-Huxley neuron model
/// Full implementation of the classic HH equations with sodium, potassium, and leak channels
#[derive(Debug, Clone, PartialEq)]
pub struct DetailedHHNeuron {
    /// Neuron identifier
    pub id: NeuronId,
    /// Current membrane potential in millivolts
    pub membrane_potential: f64,
    /// Resting potential in millivolts
    pub resting_potential: f64,
    /// Reset potential after spike in millivolts
    pub reset_potential: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Membrane capacitance in nanoFarads
    pub capacitance: f64,
    /// Sodium conductance in mS/cm²
    pub g_na: f64,
    /// Potassium conductance in mS/cm²
    pub g_k: f64,
    /// Leak conductance in mS/cm²
    pub g_l: f64,
    /// Sodium reversal potential in mV
    pub e_na: f64,
    /// Potassium reversal potential in mV
    pub e_k: f64,
    /// Leak reversal potential in mV
    pub e_l: f64,
    /// Sodium activation gate variable
    pub na_m: f64,
    /// Sodium inactivation gate variable
    pub na_h: f64,
    /// Potassium activation gate variable
    pub k_n: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
    /// Remaining refractory time
    pub refractory_timer: TimeStep,
}

impl DetailedHHNeuron {
    /// Create a DetailedHHNeuron with default parameters from spiking-networks
    pub fn from_spiking_networks_defaults(id: NeuronId) -> Self {
        Self {
            id,
            membrane_potential: -65.0, // mV
            resting_potential: -65.0,  // mV
            reset_potential: -75.0,    // mV
            threshold: -55.0,          // mV (approximate)
            capacitance: 1.0,          // nF
            g_na: 120.0,               // mS/cm²
            g_k: 36.0,                 // mS/cm²
            g_l: 0.3,                  // mS/cm²
            e_na: 50.0,                // mV
            e_k: -77.0,                // mV
            e_l: -54.4,                // mV
            na_m: 0.05,                // Initial sodium activation
            na_h: 0.6,                 // Initial sodium inactivation
            k_n: 0.32,                 // Initial potassium activation
            refractory_period: 2.0,    // ms
            refractory_timer: 0,
        }
    }

    /// Update gating variables using HH equations
    fn update_gates(&mut self, dt_ms: f64) {
        let v = self.membrane_potential;

        // Sodium activation (m)
        let alpha_m = 0.1 * (v + 40.0) / (1.0 - ((-v - 40.0) / 10.0).exp());
        let beta_m = 4.0 * ((-v - 65.0) / 18.0).exp();
        let tau_m = 1.0 / (alpha_m + beta_m);
        let m_inf = alpha_m * tau_m;
        self.na_m += (m_inf - self.na_m) * dt_ms / tau_m;

        // Sodium inactivation (h)
        let alpha_h = 0.07 * ((-v - 65.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + ((-v - 35.0) / 10.0).exp());
        let tau_h = 1.0 / (alpha_h + beta_h);
        let h_inf = alpha_h * tau_h;
        self.na_h += (h_inf - self.na_h) * dt_ms / tau_h;

        // Potassium activation (n)
        let alpha_n = 0.01 * (v + 55.0) / (1.0 - ((-v - 55.0) / 10.0).exp());
        let beta_n = 0.125 * ((-v - 65.0) / 80.0).exp();
        let tau_n = 1.0 / (alpha_n + beta_n);
        let n_inf = alpha_n * tau_n;
        self.k_n += (n_inf - self.k_n) * dt_ms / tau_n;
    }
}

impl Neuron for DetailedHHNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.refractory_timer > 0 {
            self.refractory_timer = self.refractory_timer.saturating_sub(dt);
            return;
        }

        let dt_ms = dt as f64 / 1000.0; // Convert to milliseconds

        // Update gating variables
        self.update_gates(dt_ms);

        // Calculate conductances
        let i_na = self.g_na * self.na_m.powi(3) * self.na_h * (self.membrane_potential - self.e_na);
        let i_k = self.g_k * self.k_n.powi(4) * (self.membrane_potential - self.e_k);
        let i_l = self.g_l * (self.membrane_potential - self.e_l);

        // Membrane equation: Cm * dV/dt = -I_na - I_k - I_l + I_input
        let total_current = i_na + i_k + i_l - input_current;
        let dv_dt = -total_current / self.capacitance;

        self.membrane_potential += dv_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.membrane_potential >= self.threshold {
            self.reset();
            self.refractory_timer = (self.refractory_period * 1000.0) as TimeStep;

            // Reset gating variables for next spike
            self.na_m = 0.05;
            self.na_h = 0.6;
            self.k_n = 0.32;

            // Create spike
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.membrane_potential = self.reset_potential;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

/// Detailed Izhikevich neuron model
/// Efficient model that can reproduce various firing patterns
#[derive(Debug, Clone, PartialEq)]
pub struct DetailedIzhikevichNeuron {
    /// Neuron identifier
    pub id: NeuronId,
    /// Current membrane potential in millivolts
    pub membrane_potential: f64,
    /// Recovery variable for adaptation
    pub recovery_variable: f64,
    /// Recovery time constant parameter
    pub a: f64,
    /// Recovery sensitivity parameter
    pub b: f64,
    /// Reset potential parameter
    pub c: f64,
    /// Recovery increment parameter
    pub d: f64,
    /// Spike threshold in millivolts
    pub threshold: f64,
    /// Refractory period in milliseconds
    pub refractory_period: f64,
    /// Remaining refractory time
    pub refractory_timer: TimeStep,
}

impl DetailedIzhikevichNeuron {
    /// Create a regular spiking Izhikevich neuron
    pub fn regular_spiking(id: NeuronId) -> Self {
        Self {
            id,
            membrane_potential: -65.0, // mV
            recovery_variable: -14.0,  // Initial recovery
            a: 0.02,                   // Recovery time constant
            b: 0.2,                    // Recovery sensitivity
            c: -65.0,                  // Reset potential
            d: 8.0,                    // Recovery increment
            threshold: 30.0,           // Spike threshold
            refractory_period: 2.0,    // ms
            refractory_timer: 0,
        }
    }
}

impl Neuron for DetailedIzhikevichNeuron {
    fn integrate(&mut self, input_current: f64, dt: TimeStep) {
        if self.refractory_timer > 0 {
            self.refractory_timer = self.refractory_timer.saturating_sub(dt);
            return;
        }

        let dt_ms = dt as f64 / 1000.0; // Convert to milliseconds

        // Izhikevich equations
        let v = self.membrane_potential;
        let u = self.recovery_variable;

        // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        let dv_dt = 0.04 * v * v + 5.0 * v + 140.0 - u + input_current;

        // du/dt = a*(b*v - u)
        let du_dt = self.a * (self.b * v - u);

        self.membrane_potential += dv_dt * dt_ms;
        self.recovery_variable += du_dt * dt_ms;
    }

    fn update(&mut self, _dt: TimeStep) -> Option<Spike> {
        if self.membrane_potential >= self.threshold {
            // Reset membrane potential and update recovery variable
            self.membrane_potential = self.c;
            self.recovery_variable += self.d;

            self.refractory_timer = (self.refractory_period * 1000.0) as TimeStep;

            // Create spike
            Spike::new(
                self.id.into(),
                crate::time::Time::from_nanos(0),
                1.0
            ).ok()
        } else {
            None
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    fn set_membrane_potential(&mut self, voltage: f64) {
        self.membrane_potential = voltage;
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.membrane_potential = self.c;
        self.recovery_variable += self.d;
    }

    fn id(&self) -> NeuronId {
        self.id
    }

    fn set_id(&mut self, id: NeuronId) {
        self.id = id;
    }
}

    #[test]
    fn test_detailed_lif_neuron() {
        use crate::time::TimeStepExt;
        let mut neuron = DetailedLIFNeuron::from_spiking_networks_defaults(NeuronId(0));
        let dt = TimeStep::from_ms(0.1);
        
        // Test integration
        neuron.integrate(1.0, dt);
        assert!(neuron.membrane_potential() > -75.0);
        
        // Test spiking
        neuron.set_membrane_potential(-50.0);
        let spike = neuron.update(dt);
        assert!(spike.is_some());
        assert_eq!(neuron.membrane_potential(), -75.0); // Should be reset
    }

    #[test]
    fn test_detailed_hh_neuron() {
        use crate::time::TimeStepExt;
        let mut neuron = DetailedHHNeuron::from_spiking_networks_defaults(NeuronId(0));
        let dt = TimeStep::from_ms(0.01); // Smaller timestep for HH
        
        // Test integration with sodium spike
        neuron.integrate(10.0, dt); // Strong current to trigger spike
        
        // Test that gates are updating
        assert!(neuron.na_m >= 0.0 && neuron.na_m <= 1.0);
        assert!(neuron.na_h >= 0.0 && neuron.na_h <= 1.0);
        assert!(neuron.k_n >= 0.0 && neuron.k_n <= 1.0);
    }

    #[test]
    fn test_detailed_izhikevich_neuron() {
        use crate::time::TimeStepExt;
        let mut neuron = DetailedIzhikevichNeuron::regular_spiking(NeuronId(0));
        let dt = TimeStep::from_ms(0.1);
        
        // Test integration
        neuron.integrate(10.0, dt);
        assert!(neuron.membrane_potential() > -65.0);
        
        // Test spiking
        // Izhikevich needs strong current to spike quickly
        for _ in 0..100 {
            neuron.integrate(100.0, dt);
            if let Some(_) = neuron.update(dt) {
                break;
            }
        }
        
        // Should eventually spike and reset
        neuron.set_membrane_potential(35.0);
        let spike = neuron.update(dt);
        assert!(spike.is_some());
        assert_eq!(neuron.membrane_potential(), -65.0); // Reset value
        assert!(neuron.recovery_variable > 0.0); // Recovery variable updated
    }