# Getting Started with Spiking Neural Networks Framework

This guide will help you get up and running with the spiking neural networks framework, covering installation, basic concepts, and your first simulation.

## Installation

### Primary Installation (Python)

For most users, install the Python interface:

```bash
# 1. Ensure you have Python 3.8+ and pip
python --version
pip --version

# 2. Clone the repository
git clone https://github.com/hsnn-project/hsnn.git
cd hsnn

# 3. Install the Python package
pip install maturin
maturin develop --release

# 4. Verify installation
python -c "import lixirnet as ln; print('✓ Ready to simulate!')"
```

**That's it!** This installs everything needed for basic neural network simulations.

### Advanced Installation (Rust)

For direct Rust development or advanced features:

```bash
# Install nightly Rust toolchain
rustup install nightly
rustup override set nightly

# Build from source
cargo build --release
```

## Basic Concepts

### Neurons

The framework implements several spiking neuron models:

- **Leaky Integrate-and-Fire (LIF)**: Simple model with passive membrane and fixed threshold
- **Adaptive Exponential (AdEx)**: More biologically realistic with adaptation mechanisms
- **Izhikevich**: Versatile model that can reproduce various neuronal behaviors

### Lattices and Networks

- **Lattice**: A 2D grid of neurons with local connectivity
- **Network**: Collection of lattices that can be interconnected
- **Synapses**: Connections between neurons with weights and delays

### Simulation

Simulations run in discrete time steps, processing spikes and updating neuron states according to the chosen model.

## Your First Simulation: Simple Izhikevich Lattice

Let's create a basic simulation with an Izhikevich neuron lattice.

### Python Example

```python
import lixirnet as ln
import matplotlib.pyplot as plt
import numpy as np

# Create a 5x5 lattice of Izhikevich neurons
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 5, 5)

# Connect neurons with random excitatory synapses
def connection_condition(x, y):
    # Connect if within distance 2 and not self-connection
    distance = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    return distance <= 2 and x != y

lattice.connect(connection_condition)

# Randomize initial voltages
def randomize_voltage(neuron):
    neuron.current_voltage = np.random.uniform(-65, -50)
    return neuron

lattice.apply(randomize_voltage)

# Enable history recording
lattice.update_grid_history = True

# Run simulation for 1000 time steps
lattice.run_lattice(1000)

# Plot the average membrane potential over time
history = np.array(lattice.history)
avg_potential = np.mean(history, axis=(1, 2))

plt.plot(avg_potential)
plt.xlabel('Time Steps')
plt.ylabel('Average Membrane Potential (mV)')
plt.title('Simple Izhikevich Lattice Simulation')
plt.show()
```

### Rust Example

```rust
use shnn_core::{Lattice, IzhikevichLattice, IzhikevichNeuron};
use shnn_runtime::Simulation;

fn main() {
    // Create lattice
    let mut lattice = IzhikevichLattice::new();
    lattice.populate(IzhikevichNeuron::default(), 5, 5);

    // Connect neurons (simplified - you'd implement proper connectivity)
    lattice.connect_all(0.1); // 10% connection probability

    // Run simulation
    let mut simulation = Simulation::new(lattice);
    simulation.run(1000);

    // Access results
    let spikes = simulation.get_spike_history();
    println!("Total spikes: {}", spikes.len());
}
```

## What Happens in the Simulation?

1. **Initialization**: Neurons are created with default parameters and random initial conditions
2. **Connectivity**: Synapses are established based on the connection function
3. **Simulation Loop**:
   - Each neuron integrates incoming spikes and background input
   - When membrane potential reaches threshold, a spike is generated
   - Spikes propagate to connected neurons with synaptic delay
4. **Data Collection**: Membrane potentials and spike times are recorded

## Next Steps

- Explore different neuron models in the [Basic Tutorial](basic-tutorial.md)
- Learn about plasticity and learning in the [Advanced Tutorials](advanced/)
- Check out the [Examples Gallery](examples/gallery.md) for more complex simulations

## Troubleshooting

### Common Python Issues

- **Import Error**: Ensure `maturin develop --release` completed successfully
- **Performance Issues**: Use release builds (`--release`) for better performance

### Common Rust Issues

- **Toolchain Mismatch**: Ensure you're using nightly Rust as specified
- **Compilation Errors**: Check that all dependencies are installed

For more help, see the [Basic Tutorial](basic-tutorial.md) or check existing issues in the repository.