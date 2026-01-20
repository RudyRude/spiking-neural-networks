# Spiking Neural Networks (SNN)

A unified, high-performance spiking neural network framework written in Rust, with Python bindings and comprehensive data format support.

## Features

- **Multiple Neuron Models**: LIF, AdEx, Izhikevich, Hodgkin-Huxley (coming soon)
- **Flexible Connectivity**: Hypergraph, matrix/sparse representations, plastic connections
- **NIR Compilation**: Compile networks from Neural Intermediate Representation
- **Data Formats**: Export spike data in GraphML, LPG-JSON, RDF-NQuads, and more
- **Python Bindings**: Full Python API for easy integration
- **CLI Tools**: Command-line interface for simulation and analysis
- **Modular Architecture**: Microcontroller support via `shnn-micro`

## Installation

### Rust

```bash
cargo build --release
```

### Python

```bash
pip install spiking-neural-networks
```

## Usage

### CLI

```bash
# Compile and run a NIR network
shnn nir compile network.nir -o output.json

# Export spike data in multiple formats
shnn export spikes.vmsk --format graphml
```

### Python

```python
import shnn

# Create a network
network = shnn.Network()

# Add neurons and connections
network.add_neuron(shnn.LIFNeuron())
network.connect(0, 1, weight=0.5)

# Run simulation
spikes = network.run(timesteps=1000)

# Export data
network.export_spikes("output.graphml")
```

## Architecture

The framework is organized as a Cargo workspace with multiple crates:

- `shnn-core`: Core neuron models and simulation engine
- `shnn-cli`: Command-line interface
- `shnn-python`: Python bindings
- `shnn-micro`: Microcontroller-optimized version
- `ndfh-*`: Data format crates for NDF-H support

## Neuron Models

### Implemented
- **LIF (Leaky Integrate-and-Fire)**: Basic spiking neuron model
- **AdEx (Adaptive Exponential)**: Adaptive spiking with exponential dynamics
- **Izhikevich**: Wide range of spiking patterns
- **Detailed Models**: Full implementations with biological parameters

### Coming Soon
- Hodgkin-Huxley with neurotransmission
- Astrocyte-coupled neurons

## Data Formats

Export spike data and network configurations in multiple formats:

- **GraphML**: XML-based graph format
- **LPG-JSON**: Labeled Property Graph JSON
- **RDF-NQuads**: Semantic web format
- **VEVT/ VMSK**: Binary spike formats

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Licensed under MIT License. See [LICENSE](LICENSE) for details.
