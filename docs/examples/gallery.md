# Examples Gallery

This gallery showcases various spiking neural network simulations and applications built with the framework. Each example demonstrates different concepts, from basic network dynamics to advanced cognitive models.

## Basic Examples

### Raster Plot (`raster.py`)

**Description**: Creates a basic raster plot visualization of spiking activity in a 5x5 Izhikevich neuron lattice.

**Key Features**:
- Izhikevich neuron model
- Local connectivity with distance-based weights
- Spike timing visualization
- Random initial conditions

**Code Snippet**:
```python
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 5, 5)
lattice.connect(connection_conditional)
lattice.run_lattice(10000)

# Extract spike times
peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]
```

**Output**: Raster plot showing spike times for each neuron over 10,000 time steps.

**Concepts**: Basic simulation, spike detection, visualization.

### Inhibitory-Excitatory Balance (`inh_exc.py`)

**Description**: Demonstrates balance between excitatory and inhibitory neural populations using interconnected lattices.

**Key Features**:
- Multi-lattice networks
- Excitatory-inhibitory balance
- Inter-lattice connections
- Population-level dynamics

**Code Snippet**:
```python
network = ln.IzhikevichNetwork()
network.add_lattice(exc_lattice)  # Excitatory lattice
network.add_lattice(inh_lattice)  # Inhibitory lattice
network.connect(0, 1, lambda x, y: True, lambda x, y: -1)  # Exc -> Inh (inhibitory)
network.connect(1, 0, lambda x, y: True)  # Inh -> Exc (excitatory)
```

**Output**: Time series plots of average membrane potentials for excitatory and inhibitory populations.

**Concepts**: Network architecture, E-I balance, multi-population dynamics.

### Head Direction Model (`hd_model.py`)

**Description**: Models head direction cells using ring attractor networks with velocity input.

**Key Features**:
- Ring topology
- Continuous attractor dynamics
- Velocity-driven updates
- Winner-take-all competition

**Code Snippet**:
```python
# Head direction weight function
hd_weight = lambda x, y: 3 * (np.exp(-2 * ring_distance(n, x[0], y[0]) ** 2 / (n * 10))) - 0.9

# Ring distance for periodic boundary conditions
ring_distance = lambda length, i, j: min(abs(i - j), length - abs(i - j))
```

**Output**: Activity bump moving around the ring network representing head direction.

**Concepts**: Continuous attractors, population coding, spatial navigation.

## Memory and Learning Examples

### Binary Autoassociative Network (`binary_autoassociative_network.py`)

**Description**: Implements a Hopfield network for associative memory of binary patterns.

**Key Features**:
- Hebbian learning rule
- Pattern storage and recall
- Memory capacity analysis
- Pattern completion

**Code Snippet**:
```python
def get_weights(n, patterns, a=0, b=0, scalar=1):
    w = np.zeros([n, n])
    for pattern in patterns:
        for i in range(n):
            for j in range(n):
                w[i][j] += (pattern[i] - b) * (pattern[j] - a)
    return w
```

**Output**: Recalled patterns from partial or noisy inputs.

**Concepts**: Associative memory, Hebbian learning, pattern completion.

### Synaptic Pruning in Schizophrenia (`schizophrenic_synaptic_pruning.py`)

**Description**: Models the effects of synaptic pruning on memory performance, inspired by schizophrenia research.

**Key Features**:
- Variable synaptic connectivity
- Memory recall accuracy measurement
- Distorted pattern input
- Statistical analysis across trials

**Code Snippet**:
```python
connectivities = [1, 0.8, 0.6, 0.4, 0.2]
# ... simulation loop ...
accuracy = max([acc(patterns[0], np.array([len(i) for i in peaks]), threshold=i)
                for i in range(0, firing_max)])
```

**Output**: Plot of memory accuracy vs. synaptic connectivity percentage.

**Concepts**: Neurological disorders, synaptic plasticity, memory performance.

## GPU-Accelerated Examples

### Bayesian Inference Pipeline (`bayesian_inference_pipeline.py`)

**Description**: GPU-accelerated Bayesian inference using spiking neural networks for probabilistic computation.

**Key Features**:
- CUDA/OpenCL acceleration
- Probabilistic neural coding
- Real-time inference
- Large-scale networks

**Code Snippet**:
```python
# GPU lattice creation
gpu_lattice = ln.IzhikevichNeuronLatticeGPU()
cuda_lattice.populate(ln.IzhikevichNeuron(), 100, 100)
cuda_lattice.run_lattice_gpu(10000)
```

**Output**: Probabilistic inference results with performance metrics.

**Concepts**: GPU acceleration, probabilistic computing, real-time processing.

### Grid Cell Model (Electrical) (`grid_cell_electrical_model.py`)

**Description**: Models grid cells using electrical synapses and path integration mechanisms.

**Key Features**:
- Grid cell spatial coding
- Path integration
- Electrical synapses (gap junctions)
- 2D navigation

**Code Snippet**:
```python
# Grid cell lattice with electrical synapses
lattice = ln.GridCellLattice()
lattice.set_electrical_synapses(True)
lattice.set_path_integration_velocity(vx, vy)
```

**Output**: Hexagonal grid patterns in neural activity during navigation.

**Concepts**: Spatial cognition, electrical synapses, neural coding.

### Grid Cell Model (Electrochemical) (`grid_cell_electrochemical.py`)

**Description**: Advanced grid cell model incorporating both electrical and chemical synapses with neuromodulation.

**Key Features**:
- Combined synaptic types
- Neurotransmitter modulation
- Velocity-controlled dynamics
- Multiple spatial scales

**Code Snippet**:
```python
# Electrochemical synapses
lattice.set_chemical_synapses(True)
lattice.add_neuromodulator('acetylcholine', concentration)
lattice.set_velocity_input(vx, vy, omega)
```

**Output**: Multi-scale grid patterns with velocity-dependent shifts.

**Concepts**: Neuromodulation, multi-synaptic transmission, complex spatial coding.

### Head Direction Model (GPU) (`hd_model.py` in interface_gpu)

**Description**: GPU-accelerated head direction model with real-time velocity integration.

**Key Features**:
- CUDA acceleration
- Real-time performance
- Angular path integration
- Vestibular input simulation

**Code Snippet**:
```python
# GPU-accelerated HD model (example implementation)
# gpu_hd = ln.HeadDirectionModelGPU()  # Not yet implemented
gpu_hd.set_angular_velocity(omega)
gpu_hd.integrate_motion(dt)
```

**Output**: Real-time head direction tracking with GPU performance metrics.

**Concepts**: GPU acceleration, real-time neuroscience, sensorimotor integration.

### HD Model with Electrochemical Dynamics (`hd_electrochemical_model.py`)

**Description**: Full electrochemical implementation of head direction cells with neurotransmitter dynamics.

**Key Features**:
- Glutamate/NMDA signaling
- Dendritic integration
- Plasticity mechanisms
- Turning behavior modeling

**Code Snippet**:
```python
# Electrochemical HD cell
hd_cell = ln.ElectrochemicalHDCell()
hd_cell.set_glutamate_kinetics(release_rate, reuptake_rate)
hd_cell.set_turning_input(angular_velocity)
```

**Output**: Realistic head direction signals with biochemical detail.

**Concepts**: Neurotransmitter dynamics, dendritic computation, behavioral modeling.

### Pipeline Setup (`pipeline_setup.py`)

**Description**: Demonstrates setting up complex multi-stage processing pipelines for neural computation.

**Key Features**:
- Modular pipeline architecture
- Data flow management
- Multi-stage processing
- Performance monitoring

**Code Snippet**:
```python
# Neural processing pipeline
pipeline = ln.NeuralPipeline()
pipeline.add_stage('sensory', sensory_lattice)
pipeline.add_stage('processing', processing_lattice)
pipeline.add_stage('motor', motor_lattice)
pipeline.connect_stages('sensory', 'processing')
```

**Output**: End-to-end neural processing with timing analysis.

**Concepts**: Neural architectures, data flow, modular design.

## Rate-Based Examples

### Bayesian Inference (Rate-Based) (`bayesian_inference_pipeline_rate_based.py`)

**Description**: Rate-based implementation of Bayesian inference using population coding.

**Key Features**:
- Rate coding
- Population-level computation
- Statistical inference
- Slower but interpretable dynamics

**Code Snippet**:
```python
# Rate-based inference
rate_network = ln.RateBasedNetwork()
rate_network.set_population_size(100)
rate_network.bayesian_update(likelihood, prior)
```

**Output**: Probability distributions evolving over time.

**Concepts**: Rate coding, Bayesian inference, population coding.

## Running the Examples

### Prerequisites

```bash
# Install dependencies
pip install numpy scipy matplotlib seaborn tqdm

# For GPU examples
# Install CUDA toolkit or ensure OpenCL drivers
```

### Execution

```bash
# Navigate to examples directory
cd interface/examples

# Run basic example
python raster.py

# Run GPU example
cd ../interface_gpu/experiments
python grid_cell_electrical_model.py
```

### Performance Notes

- GPU examples require compatible hardware
- Large networks may need significant memory
- Some examples include timing measurements
- Adjust parameters for different performance targets

## Extending the Examples

### Adding New Examples

1. Create new Python file in appropriate directory
2. Follow naming conventions (descriptive names)
3. Include visualization and analysis
4. Add documentation comments
5. Test with different parameter values

### Customization

```python
# Example template
import lixirnet as ln
import numpy as np
import matplotlib.pyplot as plt

# Setup
lattice = ln.IzhikevichLattice()
# ... configure lattice ...

# Simulation
lattice.run_lattice(1000)

# Analysis
# ... analyze results ...

# Visualization
plt.plot(results)
plt.show()
```

## Troubleshooting

### Common Issues

- **Memory Errors**: Reduce lattice size or disable history recording
- **GPU Errors**: Check driver versions and GPU memory
- **Slow Performance**: Use GPU acceleration or reduce simulation time
- **Import Errors**: Ensure proper installation of dependencies

### Performance Optimization

- Use GPU lattices for large simulations
- Disable history recording for long runs
- Use sparse connectivity for realistic networks
- Profile code to identify bottlenecks

## Related Documentation

- [Getting Started Guide](../getting-started.md) - Installation and basic usage
- [Basic Tutorial](../basic-tutorial.md) - Core concepts and API
- [Advanced Tutorials](../advanced/) - Specialized topics
- [API Reference](api-reference.md) - Complete API documentation