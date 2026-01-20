# SHNN Rust API Reference

This document provides comprehensive API documentation for the Rust interfaces of the SHNN (Spiking Hypergraph Neural Network) library. The library is organized into several crates, each providing specific functionality for neuromorphic computing.

## API Stability

SHNN follows semantic versioning (semver). API stability levels are indicated with badges:

- ![Stable](https://img.shields.io/badge/stability-stable-green) **Stable**: APIs that are mature and will maintain backward compatibility in future minor and patch releases
- ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) **Experimental**: APIs that may change significantly in future releases
- ![Unstable](https://img.shields.io/badge/stability-unstable-red) **Unstable**: APIs under active development with no compatibility guarantees

## Core Crates Overview

### shnn-core ![Stable](https://img.shields.io/badge/stability-stable-green)

The core crate provides fundamental neuromorphic primitives and building blocks for spiking neural networks.

#### Key Types and Structs

##### Neuron Models
- **`LIFNeuron`** - Leaky Integrate-and-Fire neuron model
- **`AdExNeuron`** - Adaptive Exponential Integrate-and-Fire neuron
- **`IzhikevichNeuron`** - Izhikevich neuron model with rich dynamics
- **`DetailedLIFNeuron`** - Detailed LIF with additional parameters
- **`DetailedHHNeuron`** - Hodgkin-Huxley neuron model
- **`DetailedIzhikevichNeuron`** - Detailed Izhikevich model

##### Data Structures
- **`Spike`** - Represents a spike event with neuron ID, timestamp, and amplitude
- **`NeuronId`** - Unique identifier for neurons (u32)
- **`Hyperedge`** - Multi-synaptic connection representation
- **`HypergraphNetwork`** - Hypergraph-based network structure

##### Plasticity
- **`STDPRule`** - Spike-Timing Dependent Plasticity rule
- **`PlasticityRule`** - Trait for plasticity mechanisms

##### Network Structures
- **`SpikeNetwork`** - Generic spiking neural network with configurable connectivity
- **`NetworkBuilder`** - Builder pattern for constructing networks
- **`NetworkConnectivity`** - Connectivity management trait

#### Key Traits

```rust
pub trait Neuron {
    fn step(&mut self, input: f32, dt: f32) -> Option<Spike>;
    fn voltage(&self) -> f32;
    fn reset(&mut self);
}

pub trait IterateAndSpike {
    fn iterate_and_spike(&mut self, inputs: &[f32], dt: f32) -> Vec<Spike>;
}

pub trait PlasticityRule {
    fn apply(&mut self, pre_time: Time, post_time: Time, delta_w: &mut f32);
}

pub trait NetworkConnectivity {
    fn connect(&mut self, source: NeuronId, target: NeuronId, weight: f32);
    fn get_connections(&self, neuron: NeuronId) -> Vec<(NeuronId, f32)>;
}
```

#### Usage Examples

##### Creating a Basic Network

```rust
use shnn_core::prelude::*;

let mut network = NetworkBuilder::new()
    .with_neurons(1000)
    .with_connectivity(0.1)
    .build::<LIFNeuron>()
    .unwrap();

// Add some spikes
let spike = Spike::new(NeuronId(0), Time::from_millis(10));
network.process_spike(spike);
```

##### Custom Neuron Implementation

```rust
use shnn_core::{Neuron, Time, Spike};

#[derive(Clone)]
pub struct CustomNeuron {
    voltage: f32,
    threshold: f32,
    tau: f32,
}

impl Neuron for CustomNeuron {
    fn step(&mut self, input: f32, dt: f32) -> Option<Spike> {
        self.voltage += (input - self.voltage) / self.tau * dt;

        if self.voltage >= self.threshold {
            self.voltage = 0.0;
            Some(Spike::new(NeuronId(0), Time::now()))
        } else {
            None
        }
    }

    fn voltage(&self) -> f32 { self.voltage }
    fn reset(&mut self) { self.voltage = 0.0; }
}
```

### shnn-async-runtime ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

Purpose-built async runtime optimized for real-time neuromorphic workloads.

#### Key Types

- **`SHNNRuntime`** - Main runtime structure
- **`RealtimeConfig`** - Configuration for real-time processing
- **`SpikeEvent`** - Spike event with nanosecond timestamp
- **`SpikeTaskHandle`** - Handle for spike processing tasks

#### Methods

```rust
impl SHNNRuntime {
    pub fn new_realtime(config: RealtimeConfig) -> Self;
    pub fn spawn_spike_task<F, T>(&self, future: F) -> SpikeTaskHandle<T>
        where F: Future<Output = T> + Send + 'static;
    pub async fn process_spike_batch(&self, spikes: &[SpikeEvent]) -> ProcessingResult;
    pub fn shutdown(&self);
}
```

#### Usage Examples

##### Real-time Spike Processing

```rust
use shnn_async_runtime::{SHNNRuntime, RealtimeConfig};

let runtime = SHNNRuntime::new_realtime(RealtimeConfig::default());

// Spawn high-priority spike task
let handle = runtime.spawn_spike_task(async {
    process_neural_spikes().await
});

// Process spike batch
let spikes = vec![
    SpikeEvent::new(1000, 42, 1.0),
    SpikeEvent::new(1500, 43, 0.8),
];

let result = runtime.process_spike_batch(&spikes).await;
println!("Processed {} spikes in {:?}", result.processed_count, result.latency);
```

### shnn-ffi

Foreign Function Interface for C/C++ interop and hardware acceleration.

#### Key Functions

```c
// Network management
int32_t hsnn_network_builder_new(HSNN_NetworkBuilder **out_builder);
int32_t hsnn_network_build(HSNN_NetworkBuilder *builder, HSNN_Network **out_network);
void hsnn_network_free(HSNN_Network *network);

// Spike processing
int32_t hsnn_run_fixed_step_vevt_consume(
    HSNN_Network **network_ptr,
    uint64_t dt_ns,
    uint64_t duration_ns,
    uint64_t seed,
    uint8_t **out_ptr,
    uintptr_t *out_len
);

// Weight management
int32_t hsnn_network_snapshot_weights(
    const HSNN_Network *network,
    HSNN_WeightTriple **out_ptr,
    uintptr_t *out_len
);
int32_t hsnn_network_apply_weight_updates(
    HSNN_Network *network,
    const HSNN_WeightTriple *updates_ptr,
    uintptr_t updates_len,
    uintptr_t *out_applied
);
```

#### Usage Examples

##### C Integration

```c
#include "shnn_ffi.h"

HSNN_NetworkBuilder *builder;
hsnn_network_builder_new(&builder);

// Add neurons and connections
hsnn_network_builder_add_neuron_range(builder, 0, 1000);
hsnn_network_builder_add_synapse_simple(builder, 0, 1, 0.5);

// Build network
HSNN_Network *network;
hsnn_network_build(builder, &network);

// Run simulation
uint8_t *output_data;
uintptr_t output_len;
hsnn_run_fixed_step_vevt_consume(&network, 1000, 1000000, 42, &output_data, &output_len);

// Cleanup
hsnn_free_buffer(output_data);
hsnn_network_free(network);
```

### shnn-storage

Binary storage formats and efficient data access patterns for neuromorphic data.

#### Key Types

- **`NeuronId`** - Neuron identifier
- **`HyperedgeId`** - Hyperedge identifier
- **`Time`** - Nanosecond-precision time representation
- **`Spike`** - Spike event structure

#### Storage Formats

- **VCSR** - Vertex Compressed Sparse Row for graph structures
- **VEVT** - Variable Event Time for temporal spike data
- **VMSK** - Variable Mask for spatial patterns

#### Traits

```rust
pub trait HypergraphStore {
    fn insert_vertex(&mut self, properties: VertexProperties) -> Result<NeuronId>;
    fn insert_hyperedge(&mut self, sources: &[NeuronId], targets: &[NeuronId]) -> Result<HyperedgeId>;
    fn get_hyperedge(&self, id: HyperedgeId) -> Option<&Hyperedge>;
}

pub trait EventStore {
    fn insert_event(&mut self, event: Event) -> Result<()>;
    fn events_in_range(&self, start: Time, end: Time) -> Vec<Event>;
}
```

#### Usage Examples

##### Storing Spike Data

```rust
use shnn_storage::{MemoryEventStore, Spike, NeuronId, Time};

let mut store = MemoryEventStore::new();

// Store spikes
let spike1 = Spike::with_amplitude(NeuronId(42), Time::from_millis(100), 1.5);
let spike2 = Spike::with_amplitude(NeuronId(43), Time::from_millis(150), 0.8);

store.insert_event(spike1.into())?;
store.insert_event(spike2.into())?;

// Query spikes in time range
let events = store.events_in_range(
    Time::from_millis(50),
    Time::from_millis(200)
);
println!("Found {} events", events.len());
```

##### Hypergraph Storage

```rust
use shnn_storage::{MemoryStore, NeuronId, VertexProperties};

let mut store = MemoryStore::new();

// Add neurons
let neuron1 = store.insert_vertex(VertexProperties {
    position: Some([0.0, 0.0, 0.0]),
    label: Some("excitatory".to_string()),
    ..Default::default()
})?;

// Create hyperedge (multi-synapse connection)
let hyperedge_id = store.insert_hyperedge(
    &[neuron1],
    &[NeuronId(100), NeuronId(101)]
)?;
```

### shnn-python

PyO3-based Python bindings providing complete access to SHNN functionality.

#### Key Classes

- **`PyNetwork`** - Python interface to spiking networks
- **`PyNeuron`** classes - LIF, AdEx, Izhikevich neuron models
- **`PySpike`** - Spike event representation
- **`PySTDPRule`** - STDP plasticity rule

#### Usage Examples

##### Basic Network Creation

```rust
use shnn_python::PyNetwork;

let network = PyNetwork::new(1000, 0.1, 0.001)?;
network.deploy_to_hardware(0)?;

let input_spikes = vec![/* spike data */];
let output = network.process_spikes(input_spikes)?;
```

## Configuration Options

### Network Configuration

```rust
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub num_neurons: u32,
    pub connectivity: f32,
    pub dt: f32,
    pub num_connections: u32,
    pub input_size: u32,
    pub output_size: u32,
    pub hidden_layers: Vec<u32>,
    pub topology: NetworkTopology,
    // ... additional fields
}
```

### Real-time Configuration

```rust
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    pub worker_count: Option<usize>,
    pub spike_buffer_size: usize,
    pub time_resolution: Duration,
    pub affinity: bool,
    pub max_task_time: Duration,
}
```

### Build Configuration

Enable features through Cargo.toml:

```toml
[dependencies.shnn-core]
version = "0.1"
features = ["async", "math", "serialize", "parallel", "hardware-accel"]
```

## Feature Flags

### shnn-core Features

- `std` - Standard library support (default)
- `no-std` - No standard library (embedded)
- `async` - Asynchronous processing
- `math` - Advanced mathematical operations
- `serialize` - Serialization support
- `simd` - SIMD optimizations
- `parallel` - Parallel processing
- `hardware-accel` - Hardware acceleration
- `zero-deps` - Minimal dependencies (default)

### shnn-async-runtime Features

- `std` - Standard library support
- `simd` - SIMD optimizations

### shnn-storage Features

- `std` - Standard library support
- `compression` - Data compression
- `encryption` - Data encryption

## Error Handling

SHNN uses a comprehensive error handling system:

```rust
use shnn_core::Result;

pub type Result<T> = std::result::Result<T, SHNNError>;

#[derive(Debug)]
pub enum SHNNError {
    InvalidNeuronId,
    NetworkNotDeployed,
    InvalidSpikeData,
    HardwareAccelerationError(String),
    // ... additional variants
}
```

## Performance Considerations

- Use `zero-deps` feature for minimal binary size
- Enable `simd` and `parallel` for performance-critical applications
- Use `shnn-async-runtime` for real-time spike processing
- Prefer `HypergraphNetwork` for complex connectivity patterns
- Use `VCSR` format for efficient graph storage

## Platform Support

- **Linux**: Full support with hardware acceleration
- **macOS**: Full support with Metal acceleration
- **Windows**: Full support with CUDA/OpenCL
- **Embedded**: no-std support for microcontrollers
- **WebAssembly**: Basic support via `shnn-wasm`