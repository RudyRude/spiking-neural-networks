# Spiking Neural Networks Rust API Reference

This document provides comprehensive API documentation for the Rust interfaces of the Spiking Neural Networks library. The library focuses on biological neuron models, plasticity, and network simulation.

## API Stability

The library follows semantic versioning (semver). API stability levels are indicated with badges:

- ![Stable](https://img.shields.io/badge/stability-stable-green) **Stable**: APIs that are mature and will maintain backward compatibility in future minor and patch releases
- ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) **Experimental**: APIs that may change significantly in future releases
- ![Unstable](https://img.shields.io/badge/stability-unstable-red) **Unstable**: APIs under active development with no compatibility guarantees

## Core Module Structure

The library is organized around the `neuron` module with submodules for different aspects of neural computation.

### Neuron Module ![Stable](https://img.shields.io/badge/stability-stable-green)

Core neuron functionality and network structures.

#### Key Types and Structs

##### Neuron Models (Stable)
- **`IzhikevichNeuron`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Izhikevich neuron model with rich dynamics
- **`LeakyIntegrateAndFire`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Basic leaky integrate-and-fire neuron
- **`QuadraticIntegrateAndFire`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Quadratic variant of integrate-and-fire
- **`AdaptiveLeakyIntegrateAndFire`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Adaptive leaky integrate-and-fire
- **`AdaptiveExponentialLeakyIntegrateAndFire`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Adaptive exponential leaky integrate-and-fire
- **`SimpleLeakyIntegrateAndFire`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Simple leaky integrate-and-fire

##### Network Structures (Stable)
- **`Lattice`** ![Stable](https://img.shields.io/badge/stability-stable-green) - 2D grid of neurons with connectivity
- **`LatticeNetwork`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Network of interconnected lattices
- **`SpikeTrainLattice`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Lattice of spike train inputs

##### Plasticity (Stable)
- **`STDP`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Spike-Timing Dependent Plasticity

##### History Types (Stable)
- **`GridVoltageHistory`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Grid voltage tracking
- **`AverageVoltageHistory`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Average voltage tracking
- **`SpikeHistory`** ![Stable](https://img.shields.io/badge/stability-stable-green) - Spike event tracking

##### Advanced Neuron Models (Experimental)
- **`HodgkinHuxleyNeuron`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Hodgkin-Huxley model with ion channels
- **`MorrisLecarNeuron`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Morris-Lecar model

##### Advanced Plasticity (Experimental)
- **`BCM`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Bienenstock-Cooper-Munro plasticity
- **`RewardModulatedSTDP`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Reward-modulated STDP
- **`TripletSTDP`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Triplet-based STDP

##### Complex Architectures (Experimental)
- **`HopfField`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Hopfield network implementation
- **`RingAttractor`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Ring attractor for spatial representations
- **`LiquidStateMachine`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Liquid state machine reservoir

#### Key Traits

```rust
/// Core iteration and spiking trait
pub trait IterateAndSpike<N: NeurotransmitterType> {
    fn iterate_and_spike(&mut self, input_current: f32) -> bool;
    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        neurotransmitter_concentrations: &NeurotransmitterConcentrations<N>
    ) -> bool;
}

/// Lattice history tracking
pub trait LatticeHistory: Default + Clone + Send + Sync {
    fn update<T: IterateAndSpike>(&mut self, state: &[Vec<T>]);
    fn reset(&mut self);
}

/// Network execution
pub trait RunNetwork {
    fn run_lattices(&mut self, iterations: usize) -> Result<(), SpikingNeuralNetworksError>;
}
```

#### Usage Examples

##### Creating a Basic Lattice

```rust
use spiking_neural_networks::neuron::{
    integrate_and_fire::IzhikevichNeuron, 
    Lattice, RunLattice, LatticeHistory, GridVoltageHistory, plasticity::STDP
};

let base_neuron = IzhikevichNeuron::default_impl();

let mut lattice: Lattice<_, _, GridVoltageHistory, STDP, _> = Lattice::default();
lattice.populate(&base_neuron, 10, 10)?;
lattice.connect(&|x, y| x != y, Some(&|_, _| 0.5));

// Run simulation
lattice.run_lattice(1000)?;
```

##### Custom Neuron Implementation

```rust
use spiking_neural_networks::neuron::iterate_and_spike::{
    IterateAndSpike, NeurotransmitterType, IonotropicNeurotransmitterType
};

#[derive(Clone)]
pub struct CustomNeuron {
    pub current_voltage: f32,
    pub v_th: f32,
}

impl IterateAndSpike for CustomNeuron {
    type N = IonotropicNeurotransmitterType;

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        self.current_voltage += input_current;
        if self.current_voltage >= self.v_th {
            self.current_voltage = 0.0;
            true
        } else {
            false
        }
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        _: &HashMap<Self::N, f32>
    ) -> bool {
        self.iterate_and_spike(input_current)
    }
}
```

### GPU Support ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

GPU-accelerated lattice computations using OpenCL.

#### Key Types

- **`LatticeGPU`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - GPU-accelerated lattice
- **`LatticeNetworkGPU`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - GPU network

#### Usage Example

```rust
#[cfg(feature = "gpu")]
use spiking_neural_networks::neuron::gpu_lattices::LatticeGPU;

#[cfg(feature = "gpu")]
{
    let gpu_lattice = LatticeGPU::from_cpu_lattice(&cpu_lattice)?;
    gpu_lattice.run_lattice_gpu(1000)?;
}
```

### Digital Twin ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

Modular brain simulation framework.

#### Key Types

- **`BrainRegion`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Trait for brain regions
- **`DigitalTwin`** ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) - Orchestrator for brain simulation

## Configuration Options

### Lattice Configuration

```rust
impl<T, U, V, W, N> Lattice<T, U, V, W, N> 
where
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32>,
    V: LatticeHistory,
    W: Plasticity<T, T, f32>,
    N: NeurotransmitterType,
{
    pub fn set_dt(&mut self, dt: f32);
    pub fn set_electrical_synapse(&mut self, enabled: bool);
    pub fn set_chemical_synapse(&mut self, enabled: bool);
    pub fn set_parallel(&mut self, enabled: bool);
}
```

## Error Handling

```rust
use spiking_neural_networks::error::SpikingNeuralNetworksError;

pub type Result<T> = std::result::Result<T, SpikingNeuralNetworksError>;

#[derive(Debug)]
pub enum SpikingNeuralNetworksError {
    GraphError(GraphError),
    LatticeNetworkError(LatticeNetworkError),
    AgentError(AgentError),
    // ... additional variants
}
```

## Performance Considerations

- Use parallel execution for large lattices: `lattice.parallel = true`
- Pre-allocate history structures
- Use appropriate timestep (`dt`) for numerical stability
- Consider GPU acceleration for compute-intensive simulations

## Platform Support

- **Linux**: Full support with OpenCL GPU acceleration
- **macOS**: Full support with OpenCL GPU acceleration
- **Windows**: Full support with OpenCL GPU acceleration
- **Embedded**: Limited support (no GPU)