# Advanced Usage Guide

This guide covers advanced features and techniques for using HSNN beyond basic network creation and simulation. Topics include custom plasticity rules, neuromodulation, digital twin construction, and performance optimization.

## Table of Contents

- [Custom Plasticity Rules](#custom-plasticity-rules)
- [Neuromodulation and Pathology](#neuromodulation-and-pathology)
- [Digital Twin Construction](#digital-twin-construction)
- [Performance Optimization](#performance-optimization)
- [GPU Acceleration](#gpu-acceleration)
- [Custom Neuron Models](#custom-neuron-models)

## Custom Plasticity Rules

HSNN's trait-based plasticity system allows implementing custom learning rules.

### Implementing a Custom STDP Rule

```rust
use shnn_core::plasticity::{STDPRule, PlasticityRule};

pub struct CustomSTDPRule {
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
}

impl PlasticityRule for CustomSTDPRule {
    fn apply(&mut self, pre_time: Time, post_time: Time) -> f32 {
        let delta_t = post_time - pre_time;
        if delta_t > 0.0 {
            // LTP: pre before post
            self.a_plus * (-delta_t / self.tau_plus).exp()
        } else {
            // LTD: post before pre
            -self.a_minus * (delta_t / self.tau_minus).exp()
        }
    }
}
```

### Reward-Modulated Plasticity

```python
import shnn as nn

# Create reward-modulated STDP
rule = nn.RewardModulatedSTDP(
    a_plus=0.01,
    a_minus=0.01,
    dopamine_decay=0.9,
    reward_function=my_reward_fn
)

network = nn.Network(num_neurons=1000)
network.add_plasticity_rule(rule)
```

## Neuromodulation and Pathology

### Simulating Dopamine Effects

```python
# Create dopamine-modulated network
dopamine_system = nn.DopamineSystem(
    baseline_concentration=0.1,
    decay_rate=0.95
)

network = nn.Network(num_neurons=500)
network.add_neuromodulator(dopamine_system)

# Trigger reward event
network.deliver_reward(reward_value=1.0)
```

### Pathology Simulation

```python
# Simulate schizophrenia (NMDA hypofunction)
pathology = nn.PathologyModel.schizophrenia(
    nmda_reduction=0.3,
    gaba_imbalance=0.2
)

network = nn.Network(num_neurons=1000)
network.apply_pathology(pathology)

# Run simulation and analyze hallucinations
results = network.simulate(duration=10.0)
hallucination_events = results.detect_abnormal_patterns()
```

## Digital Twin Construction

### Building a Multi-Region Brain Model

```python
# Create brain regions
cortex = nn.CorticalModule(
    neurons=2000,
    plasticity=nn.STDP(),
    connectivity=nn.LocalConnectivity(radius=3)
)

hippocampus = nn.HippocampalModule(
    neurons=500,
    attractor_type=nn.RingAttractor(),
    head_direction_input=True
)

# Assemble digital twin
brain = nn.DigitalTwin()
brain.add_region("cortex", cortex)
brain.add_region("hippocampus", hippocampus)

# Connect regions
brain.connect_regions(
    "cortex", "hippocampus",
    connection_type=nn.ChemicalSynapse(strength=0.5)
)

# Run multi-region simulation
brain.simulate(duration=5.0, dt=0.001)
```

### Custom Brain Region

```rust
use shnn_core::digital_twin::{BrainRegion, DigitalTwin};

pub struct CustomRegion {
    neurons: Vec<IzhikevichNeuron>,
    // ... other fields
}

impl BrainRegion for CustomRegion {
    fn step(&mut self, inputs: &[f32], dt: f32) -> Vec<Spike> {
        // Custom region logic
        // ...
    }

    fn get_output(&self) -> &[f32] {
        // Return region output
        // ...
    }
}
```

## Performance Optimization

### Memory Pre-allocation

```rust
// Pre-allocate spike buffers
network = nn.Network(
    num_neurons=10000,
    spike_buffer_size=100000,  # Pre-allocate for 100k spikes
    weight_matrix_prealloc=True
)
```

### Parallel Processing Configuration

```python
# Optimize for multi-core
config = nn.SimulationConfig(
    parallel_processing=True,
    num_threads=8,
    chunk_size=1000,  # Process neurons in chunks
    cache_line_optimization=True
)

results = network.simulate_with_config(duration=10.0, config=config)
```

### Profiling and Benchmarking

```python
# Enable performance profiling
profiler = nn.Profiler()
profiler.start()

results = network.simulate(duration=1.0)

# Analyze bottlenecks
report = profiler.generate_report()
print(f"Spike processing time: {report.spike_time_ms} ms")
print(f"Memory usage: {report.memory_mb} MB")
```

## GPU Acceleration

### CUDA Setup

```bash
# Install CUDA toolkit
# Build with CUDA support
cargo build --release --features cuda
```

### GPU Network Configuration

```python
# Create GPU-accelerated network
network = nn.Network(num_neurons=50000)

# Deploy to GPU
accelerator = nn.CUDAAccelerator(device_id=0)
network.deploy_to_accelerator(accelerator)

# Run GPU simulation
results = network.simulate_gpu(duration=10.0, streams=4)
```

### OpenCL Support

```python
# AMD/Intel GPU acceleration
accelerator = nn.OpenCLAccelerator(platform_id=0, device_id=0)
network.deploy_to_accelerator(accelerator)
```

## Custom Neuron Models

### Implementing Custom Dynamics

```rust
use shnn_core::neuron::{Neuron, IterateAndSpike};

pub struct CustomNeuron {
    voltage: f32,
    adaptation: f32,
    // Custom parameters
    custom_param: f32,
}

impl Neuron for CustomNeuron {
    fn step(&mut self, input: f32, dt: f32) -> Option<Spike> {
        // Custom dynamics
        self.voltage += (input - self.voltage) * dt / self.tau;
        self.adaptation *= 0.99;  // Decay adaptation

        if self.voltage > self.threshold {
            self.voltage = self.reset;
            self.adaptation += self.adaptation_increment;
            Some(Spike::new(self.id, Time::now()))
        } else {
            None
        }
    }

    fn voltage(&self) -> f32 { self.voltage }
    fn reset(&mut self) { self.voltage = self.reset; }
}
```

### Loading from .nb Files

```python
# Load custom neuron model
model_spec = nn.load_neuron_model("my_model.nb")

network = nn.Network(num_neurons=1000)
network.set_neuron_model(model_spec)

# The model is now used for all neurons
results = network.simulate(duration=5.0)
```

### Parameter Optimization

```python
# Optimize neuron parameters
optimizer = nn.ParameterOptimizer(
    network=network,
    target_metric=nn.FiringRate(target_rate=10.0),
    parameters=["tau_m", "threshold", "reset"]
)

# Run optimization
best_params = optimizer.optimize(
    generations=100,
    population_size=50,
    mutation_rate=0.1
)

network.update_parameters(best_params)