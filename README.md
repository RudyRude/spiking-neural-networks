# HSNN: High-Performance Spiking Neural Network Framework

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://docs.rs/hsnn)
[![Crates.io](https://img.shields.io/crates/v/hsnn)](https://crates.io/crates/hsnn)

A high-performance, modular spiking neural network framework with pluggable connectivity structures, written in Rust with Python bindings. Designed for neuroscience research, cognitive modeling, and neuromorphic computing applications.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

### 🧠 **Neuron Models**
- **Izhikevich Model**: Versatile model reproducing various neural behaviors
- **Morris-Lecar Model**: Biologically realistic with calcium and potassium dynamics
- **Hodgkin-Huxley Model**: Detailed ion channel modeling with neurotransmission
- **Leaky Integrate-and-Fire (LIF)**: Simple and efficient for large-scale simulations
- **Custom Models**: Extensible trait system for implementing new neuron types

#### Neuron Model Breakdowns

**Leaky Integrate-and-Fire (LIF)**: Simplest spiking model balancing efficiency and dynamics. Membrane potential integrates input current with exponential decay. Variants include adaptive versions that model spike-frequency adaptation.

**Izhikevich Model**: Quadratic integrate-and-fire with recovery variable. Reproduces 20+ spiking patterns (regular, bursting, chattering) with minimal parameters. Used for cortical neuron simulation and lattice networks.

**Hodgkin-Huxley Model**: Biophysical gold standard with detailed ion channel kinetics. Models sodium, potassium, and leak currents with voltage-dependent gating. Includes full neurotransmission for synaptic modeling.

#### Kinetics Rationale

HSNN implements neurotransmitter kinetics with biological accuracy while optimizing for performance. Receptor binding uses simplified exponential kinetics (rather than complex Markov chains) to balance detail with speed. Izhikevich models use fixed receptor kinetics to maintain computational efficiency (100x+ speedup vs. full Hodgkin-Huxley) while preserving core synaptic mechanisms. This design enables large-scale cognitive simulations without sacrificing essential biological plausibility.

### 🏗️ **Network Architectures**
- **Lattice Networks**: 2D grid structures with local connectivity
- **Hypergraph Networks**: Advanced connectivity with hierarchical structures
- **Multi-Population Networks**: Interconnected neural populations
- **Digital Twin Framework**: Modular brain regions (cortex, hippocampus, LSM) with inter-connectivity
- **Custom Topologies**: Flexible graph-based connectivity

### ⚡ **Synaptic Dynamics**
- **Electrical Synapses**: Gap junction-based communication (default)
- **Chemical Synapses**: Neurotransmitter-mediated signaling
- **Spike-Timing-Dependent Plasticity (STDP)**: Learning and adaptation
- **Short/Long-Term Plasticity**: Various learning rules
- **Neuromodulation**: Dopamine, acetylcholine, and custom modulators

### 🚀 **Performance & Acceleration**
- **GPU Support**: CUDA and OpenCL acceleration for large-scale simulations
- **Parallel Processing**: Multi-threaded execution with Rayon
- **Memory Efficiency**: Lock-free data structures and optimized memory layouts
- **WebAssembly**: Browser-based simulations via WASM
- **Embedded Support**: Cortex-M microcontroller compatibility

### 🐍 **Interfaces**
- **Python API**: User-friendly Python bindings (`lixirnet`)
- **Rust API**: Direct access to high-performance Rust implementation
- **FFI Bindings**: C-compatible foreign function interface

### 📊 **Analysis & Visualization**
- **Built-in Analysis**: Spike timing, rate analysis, population dynamics
- **Visualization Tools**: Raster plots, membrane potential traces, connectivity graphs
- **Data Export**: CSV, HDF5, and custom formats
- **Real-time Monitoring**: Performance profiling and debugging tools

## Prerequisites

### Primary Requirements (for Python interface)
- **Python**: 3.8 or higher
- **pip**: Latest version (usually included with Python)
- **Git**: For cloning the repository

### Optional Requirements
- **CUDA Toolkit**: 11.0+ (for NVIDIA GPU acceleration)
- **OpenCL**: 2.0+ (for GPU acceleration on other hardware)
- **Rust**: Nightly toolchain (for direct Rust development)

## Quick Start

### Python (Recommended for most users)

```bash
# Install dependencies
pip install maturin numpy scipy matplotlib

# Clone and build
git clone https://github.com/hsnn-project/hsnn.git
cd hsnn
maturin develop --release
```

```python
import lixirnet as ln
import numpy as np

# Create a simple Izhikevich neuron lattice
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 10, 10)

# Connect neurons with local connectivity
def local_connection(x, y):
    distance = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    return distance <= 2 and x != y

lattice.connect(local_connection)

# Run simulation
lattice.run_lattice(1000)

# Analyze results
spikes = lattice.get_spike_times()
print(f"Total spikes: {len(spikes)}")
```

### Rust

```rust
use shnn_core::{Lattice, IzhikevichLattice, IzhikevichNeuron};
use shnn_runtime::Simulation;

fn main() {
    // Create lattice
    let mut lattice = IzhikevichLattice::new();
    lattice.populate(IzhikevichNeuron::default(), 10, 10);

    // Connect neurons
    lattice.connect_local(2.0); // Connect within radius 2.0

    // Run simulation
    let mut sim = Simulation::new(lattice);
    sim.run(1000);

    // Get results
    let spike_count = sim.spike_count();
    println!("Total spikes: {}", spike_count);
}
```

## Installation

### Primary Installation (Python)

For most users, especially scientific researchers, we recommend the Python interface:

```bash
# Clone the repository
git clone https://github.com/hsnn-project/hsnn.git
cd hsnn

# Install the Python package
pip install maturin
maturin develop --release
```

Verify installation:
```python
import lixirnet as ln
print("Installation successful!")
```

### Advanced Installation

#### Rust Development

For contributors or those needing direct Rust access:

```bash
# Install nightly Rust toolchain
rustup install nightly
rustup override set nightly

# Build from source
cargo build --release

# Run tests
cargo test
```

#### GPU Support

For GPU-accelerated simulations:

**NVIDIA (CUDA):**
```bash
# Install CUDA toolkit from NVIDIA website
# Then build with CUDA support
cd backend
cargo build --release --features cuda
```

**AMD/Intel (OpenCL):**
```bash
# Install OpenCL drivers for your hardware
# Build with OpenCL support
cd backend
cargo build --release --features opencl
```

#### Docker

Containerized installation:

```bash
# Build and run
docker build -t hsnn .
docker run --gpus all -it hsnn
```

### Troubleshooting

**Common Issues:**

- **Maturin fails**: Ensure Python 3.8+ and pip are installed
- **Import error**: Run `maturin develop --release` again
- **GPU not detected**: Check driver installation and feature flags
- **Compilation errors**: Ensure nightly Rust is active

For detailed troubleshooting, see [INSTALL.md](INSTALL.md).

## Documentation

- [**Getting Started**](docs/getting-started.md) - Installation and basic concepts
- [**Basic Tutorial**](docs/basic-tutorial.md) - Core API and first simulations
- [**Advanced Tutorials**](docs/advanced/)
  - [Synaptic Plasticity](docs/advanced/plasticity.md)
  - [Neurotransmitter Dynamics](docs/advanced/neurotransmitter-dynamics.md)
  - [GPU Usage](docs/advanced/gpu-usage.md)
  - [Custom Neuron Models](docs/advanced/custom-neuron-models.md)
- [**API Reference**](docs/api/)
  - [Python API](docs/api/python-api.md)
  - [Rust API](docs/api/rust-api.md)
- [**Examples Gallery**](docs/examples/gallery.md) - Code examples and applications

## Examples

### Basic Raster Plot
[![Izhikevich Raster Plot](https://github.com/hsnn-project/hsnn/blob/main/images/raster_example.png?raw=true)](docs/examples/gallery.md#raster-plot)

### Head Direction Model
[![Head Direction Model](https://github.com/hsnn-project/hsnn/blob/main/images/hd_model.png?raw=true)](docs/examples/gallery.md#head-direction-model)

### GPU-Accelerated Grid Cells
[![Grid Cell Model](https://github.com/hsnn-project/hsnn/blob/main/images/grid_cells.png?raw=true)](docs/examples/gallery.md#grid-cell-model-electrical)

## Applications

- **Neuroscience Research**: Neural dynamics, synaptic plasticity, neural coding
- **Cognitive Modeling**: Memory systems, spatial navigation, decision making
- **Machine Learning**: STDP-based classifiers, R-STDP regressors, LSM-based cognition
- **Neuromorphic Computing**: Hardware acceleration, energy-efficient computing
- **AI Research**: Spiking neural networks, brain-inspired algorithms
- **Robotics**: Sensory processing, motor control, adaptive behavior

## Performance

| Network Size | CPU (ms/step) | GPU (ms/step) | Speedup |
|-------------|----------------|----------------|---------|
| 1K neurons  | 0.5           | 0.05         | 10x    |
| 10K neurons | 5.0           | 0.2          | 25x    |
| 100K neurons| 50.0          | 1.0          | 50x    |

*Benchmarks on Intel i7-9750H + NVIDIA RTX 2070*

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/hsnn.git
cd hsnn

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
cargo test
python -m pytest interface/

# Build documentation
mkdocs build
```

### Code Style

- **Rust**: Follow standard Rust formatting (`cargo fmt`)
- **Python**: Black formatter, isort for imports
- **Documentation**: Clear, concise, with examples

## Related Projects

- [Brian2](https://brian2.readthedocs.io/) - Python spiking neural network simulator
- [NEST](https://www.nest-simulator.org/) - Neural simulation tool
- [NEURON](https://neuron.yale.edu/) - Detailed neuronal modeling
- [ Norse](https://github.com/norse/norse) - PyTorch-based spiking neural networks

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

- Built with [Rust](https://www.rust-lang.org/) for performance
- Python bindings via [PyO3](https://github.com/PyO3/pyo3) and [Maturin](https://github.com/PyO3/maturin)
- GPU acceleration using [CUDA](https://developer.nvidia.com/cuda-toolkit) and [OpenCL](https://www.khronos.org/opencl/)
- Inspired by biological neural systems and computational neuroscience research

---

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/hsnn-project/hsnn).
