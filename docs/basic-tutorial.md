# Basic Tutorial: Building Spiking Neural Networks

This tutorial covers the fundamental building blocks of spiking neural networks in this framework: neurons, lattices, networks, and simulations.

## Creating Neurons

The framework supports multiple neuron models. Let's explore each one.

### Leaky Integrate-and-Fire (LIF) Neurons

LIF neurons are the simplest model, integrating input until reaching a threshold.

```python
import lixirnet as ln

# Create a single LIF neuron with default parameters
lif_neuron = ln.LIFNeuron()

# Customize parameters
lif_neuron = ln.LIFNeuron(
    v_rest=-65.0,      # Resting potential (mV)
    v_th=-50.0,        # Threshold potential (mV)
    v_reset=-70.0,     # Reset potential (mV)
    tau_m=20.0,        # Membrane time constant (ms)
    r_m=10.0           # Membrane resistance (MΩ)
)
```

### Izhikevich Neurons

Izhikevich neurons can reproduce various firing patterns found in biological neurons.

```python
# Regular spiking neuron (cortical pyramidal cell)
rs_neuron = ln.IzhikevichNeuron(
    a=0.02, b=0.2, c=-65.0, d=8.0
)

# Fast spiking neuron (inhibitory interneuron)
fs_neuron = ln.IzhikevichNeuron(
    a=0.1, b=0.2, c=-65.0, d=2.0
)

# Chattering neuron
ch_neuron = ln.IzhikevichNeuron(
    a=0.02, b=0.2, c=-50.0, d=2.0
)
```

### Adaptive Exponential (AdEx) Neurons

AdEx neurons include adaptation mechanisms for more realistic behavior.

```python
adex_neuron = ln.AdExNeuron(
    v_rest=-70.0,      # Resting potential
    v_th=-50.0,        # Threshold
    delta_t=2.0,       # Slope factor
    a=2.0,            # Adaptation time constant
    b=0.0,            # Adaptation current
    tau_w=30.0        # Adaptation time constant
)
```

## Creating Lattices

Lattices are 2D grids of neurons with local connectivity.

### Basic Lattice Creation

```python
# Create a 10x10 lattice of Izhikevich neurons
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 10, 10)

print(f"Lattice size: {lattice.width} x {lattice.height}")
print(f"Total neurons: {lattice.size}")
```

### Different Lattice Types

```python
# LIF lattice
lif_lattice = ln.LIFLattice()
lif_lattice.populate(ln.LIFNeuron(), 8, 8)

# AdEx lattice
adex_lattice = ln.AdExLattice()
adex_lattice.populate(ln.AdExNeuron(), 8, 8)
```

## Connecting Neurons

Connections define how neurons communicate through synapses.

### Simple All-to-All Connectivity

```python
# Connect all neurons with 20% probability and default weight
lattice.connect(lambda x, y: True, weight_func=lambda x, y: 0.2)
```

### Distance-Based Connectivity

```python
import numpy as np

def distance_connection(pos1, pos2):
    """Connect neurons within distance 3"""
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance <= 3 and pos1 != pos2

def gaussian_weight(pos1, pos2):
    """Gaussian weight decay with distance"""
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return np.exp(-distance**2 / 4.0)

lattice.connect(distance_connection, gaussian_weight)
```

### Inhibitory Connections

```python
# Create inhibitory connections (negative weights)
def inhibitory_connection(pos1, pos2):
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance <= 2

lattice.connect(inhibitory_connection, lambda x, y: -0.5)
```

## Creating Networks of Lattices

Networks allow connecting multiple lattices together.

```python
# Create excitatory and inhibitory lattices
exc_lattice = ln.IzhikevichLattice(0)  # ID 0
exc_lattice.populate(ln.IzhikevichNeuron(), 10, 10)
exc_lattice.connect(distance_connection)

inh_lattice = ln.IzhikevichLattice(1)  # ID 1
inh_lattice.populate(ln.IzhikevichNeuron(), 5, 5)
inh_lattice.connect(distance_connection, lambda x, y: -1.0)  # Inhibitory

# Create network
network = ln.IzhikevichNetwork()
network.add_lattice(exc_lattice)
network.add_lattice(inh_lattice)

# Connect lattices (excitatory to inhibitory)
network.connect(0, 1, lambda x, y: True, lambda x, y: 0.3)
# Connect lattices (inhibitory to excitatory)
network.connect(1, 0, lambda x, y: True, lambda x, y: -0.2)
```

## Running Simulations

### Basic Lattice Simulation

```python
import matplotlib.pyplot as plt

# Setup lattice
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 5, 5)
lattice.connect(distance_connection)

# Enable history recording
lattice.update_grid_history = True

# Reset timing and history
lattice.reset_timing()
lattice.reset_history()

# Run simulation
lattice.run_lattice(1000)

# Analyze results
history = lattice.history  # List of 2D arrays (time x neurons)
spike_times = lattice.spike_times

print(f"Simulation ran for {len(history)} time steps")
print(f"Total spikes: {len(spike_times)}")
```

### Network Simulation

```python
# Setup network as above...

# Run network simulation
for step in range(1000):
    network.run_lattices(1)  # Run one step at a time

# Get results from individual lattices
exc_history = network.get_lattice(0).history
inh_history = network.get_lattice(1).history
```

## Visualizing Results

### Raster Plot

```python
import scipy.signal
import numpy as np

def create_raster_plot(lattice, threshold=30):
    """Create raster plot from lattice history"""
    data = np.array([np.array(step).flatten() for step in lattice.history])

    spike_times = []
    neuron_ids = []

    for neuron_id in range(data.shape[1]):
        peaks, _ = scipy.signal.find_peaks(data[:, neuron_id])
        spike_times.extend(peaks)
        neuron_ids.extend([neuron_id] * len(peaks))

    plt.scatter(spike_times, neuron_ids, s=1, c='black')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron ID')
    plt.title('Spike Raster Plot')
    plt.show()

create_raster_plot(lattice)
```

### Membrane Potential Traces

```python
def plot_membrane_traces(lattice, neuron_indices=None):
    """Plot membrane potential traces for selected neurons"""
    if neuron_indices is None:
        neuron_indices = [0, 5, 10, 15, 20]  # Sample neurons

    history = np.array([np.array(step).flatten() for step in lattice.history])

    for idx in neuron_indices:
        if idx < history.shape[1]:
            plt.plot(history[:, idx], label=f'Neuron {idx}')

    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Membrane Potential Traces')
    plt.legend()
    plt.show()

plot_membrane_traces(lattice)
```

## Customizing Neuron Parameters

### Applying Functions to Neurons

```python
def setup_neuron(neuron):
    """Initialize neuron with custom parameters"""
    neuron.current_voltage = np.random.uniform(-70, -50)
    neuron.adaptation_current = 0.0
    return neuron

# Apply to all neurons in lattice
lattice.apply(setup_neuron)
```

### Parameter Sweeps

```python
# Test different threshold values
thresholds = [-55, -50, -45, -40]

for threshold in thresholds:
    # Create fresh lattice
    test_lattice = ln.IzhikevichLattice()
    test_lattice.populate(ln.IzhikevichNeuron(v_th=threshold), 5, 5)
    test_lattice.connect(distance_connection)

    # Run simulation
    test_lattice.run_lattice(500)

    spike_count = len(test_lattice.spike_times)
    print(f"Threshold {threshold}mV: {spike_count} spikes")
```

## Best Practices

1. **Start Small**: Begin with small lattices (5x5) to test connectivity and parameters
2. **Monitor Performance**: Large simulations can be computationally expensive
3. **Use History Wisely**: Recording full history uses significant memory
4. **Parameter Tuning**: Biological ranges are a good starting point, but may need adjustment
5. **Random Seeds**: Set random seeds for reproducible results

## Next Steps

- Learn about synaptic plasticity in [Plasticity Tutorial](advanced/plasticity.md)
- Explore GPU acceleration in [GPU Usage Tutorial](advanced/gpu-usage.md)
- See complex examples in the [Examples Gallery](examples/gallery.md)