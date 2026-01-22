# Spiking Neural Networks Python API Reference

This document provides comprehensive API documentation for the Python interfaces of the Spiking Neural Networks library. The library provides CPU and GPU accelerated interfaces for biological neural network simulation.

## API Stability

The library follows semantic versioning (semver). API stability levels are indicated with badges:

- ![Stable](https://img.shields.io/badge/stability-stable-green) **Stable**: APIs that are mature and will maintain backward compatibility in future minor and patch releases
- ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) **Experimental**: APIs that may change significantly in future releases
- ![Unstable](https://img.shields.io/badge/stability-unstable-red) **Unstable**: APIs under active development with no compatibility guarantees

## Core Classes

### CPU Interface (lixirnet) ![Stable](https://img.shields.io/badge/stability-stable-green)

The CPU interface provides basic neural network simulation capabilities.

#### Neuron Models ![Stable](https://img.shields.io/badge/stability-stable-green)

```python
class PyIzhikevichNeuron:
    """Izhikevich neuron model with configurable parameters."""

    def __init__(self, a=0.02, b=0.2, c=-55., d=8., v_th=30., dt=0.1, current_voltage=-65., w_value=30., gap_conductance=10., tau_m=1., c_m=100.):
        """Initialize Izhikevich neuron.

        Args:
            a, b, c, d: Izhikevich parameters
            v_th: Threshold voltage
            dt: Time step
            current_voltage: Initial voltage
            w_value: Recovery variable
            gap_conductance: Gap junction conductance
            tau_m: Membrane time constant
            c_m: Membrane capacitance
        """

    def iterate_and_spike(self, input_current: float) -> bool:
        """Perform one iteration and return spiking status."""

    def iterate_with_neurotransmitter_and_spike(self, input_current: float, neurotransmitter_concs: Dict[str, float]) -> bool:
        """Perform iteration with neurotransmitter input."""

    @property
    def current_voltage(self) -> float: ...

    @property
    def is_spiking(self) -> bool: ...

class PyDopaIzhikevichNeuron:
    """Dopamine-modulated Izhikevich neuron with neurotransmitters."""

    def __init__(self, **kwargs):
        """Initialize dopamine-modulated neuron."""

    # Similar methods to PyIzhikevichNeuron
```

#### Network Structures ![Stable](https://img.shields.io/badge/stability-stable-green)

```python
class PyIzhikevichNeuronLattice:
    """2D lattice of Izhikevich neurons."""

    def __init__(self, lattice: Lattice):
        """Initialize lattice."""

    def run_lattice(self, iterations: int):
        """Run lattice simulation for given iterations."""

    def set_dt(self, dt: float):
        """Set time step."""

    def populate(self, base_neuron: PyIzhikevichNeuron, rows: int, cols: int):
        """Populate lattice with neurons."""

    def connect(self, condition, weights=None):
        """Connect neurons based on condition."""

    @property
    def grid_history(self):
        """Access voltage history."""

class PyIzhikevichNeuronNetwork:
    """Network of interconnected lattices."""

    def __init__(self, network: LatticeNetwork):
        """Initialize network."""

    def add_lattice(self, lattice: PyIzhikevichNeuronLattice):
        """Add lattice to network."""

    def connect(self, from_id: int, to_id: int, condition, weights=None):
        """Connect lattices."""

    def run_lattices(self, iterations: int):
        """Run network simulation."""
```

#### Plasticity ![Stable](https://img.shields.io/badge/stability-stable-green)

```python
class PySTDP:
    """Spike-Timing Dependent Plasticity."""

    def __init__(self):
        """Initialize STDP with default parameters."""

    @property
    def a_plus(self) -> float: ...
    @a_plus.setter
    def a_plus(self, value: float): ...

    @property
    def a_minus(self) -> float: ...
    @a_minus.setter
    def a_minus(self, value: float): ...

    @property
    def tau_plus(self) -> float: ...
    @tau_plus.setter
    def tau_plus(self, value: float): ...

    @property
    def tau_minus(self) -> float: ...
    @tau_minus.setter
    def tau_minus(self, value: float): ...
```

### GPU Interface (janky) ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

The GPU interface provides OpenCL-accelerated neural network simulation.

#### GPU Neuron Models ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

```python
class PyIzhikevichNeuronLatticeGPU:
    """GPU-accelerated Izhikevich neuron lattice."""

    def __init__(self, lattice: LatticeGPU):
        """Initialize GPU lattice."""

    def run_lattice_gpu(self, iterations: int):
        """Run GPU-accelerated simulation."""

    # Similar methods to CPU version

class PyIzhikevichNeuronNetworkGPU:
    """GPU-accelerated neural network."""

    def __init__(self, network: LatticeNetworkGPU):
        """Initialize GPU network."""

    def run_lattices(self, iterations: int):
        """Run GPU network simulation."""

    # Similar methods to CPU version
```

#### Spike Trains ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

```python
class PyRateSpikeTrain:
    """Rate-based spike train generator."""

    def __init__(self):
        """Initialize spike train."""

    @property
    def rate(self) -> float: ...
    @rate.setter
    def rate(self, value: float): ...

    def iterate(self) -> bool:
        """Generate next spike."""

    @property
    def is_spiking(self) -> bool: ...

class PyRateSpikeTrainLattice:
    """Lattice of rate-based spike trains."""

    def __init__(self, lattice: SpikeTrainLattice):
        """Initialize spike train lattice."""

    def populate(self, spike_train: PyRateSpikeTrain, rows: int, cols: int):
        """Populate lattice."""

    def run_lattice(self, iterations: int):
        """Run spike train simulation."""
```

## Usage Examples

### Basic CPU Simulation ![Stable](https://img.shields.io/badge/stability-stable-green)

```python
import lixirnet

# Create neuron
neuron = lixirnet.PyIzhikevichNeuron()

# Create lattice
lattice = lixirnet.PyIzhikevichNeuronLattice()
base_neuron = lixirnet.PyIzhikevichNeuron()
lattice.populate(base_neuron, 10, 10)
lattice.connect(lambda x, y: x != y, lambda _, __: 0.5)

# Run simulation
lattice.run_lattice(1000)

# Access results
history = lattice.grid_history.history
```

### GPU Simulation ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

```python
import janky

# Create GPU lattice
gpu_lattice = janky.PyIzhikevichNeuronLatticeGPU()
# ... similar setup to CPU version

# Run GPU simulation
gpu_lattice.run_lattice_gpu(1000)
```

### Plasticity Example ![Stable](https://img.shields.io/badge/stability-stable-green)

```python
import lixirnet

# Create STDP plasticity
stdp = lixirnet.PySTDP()
stdp.a_plus = 0.01
stdp.a_minus = 0.01
stdp.tau_plus = 20.0
stdp.tau_minus = 20.0

# Create lattice with plasticity
lattice = lixirnet.PyIzhikevichNeuronLattice()
lattice.plasticity = stdp
lattice.do_plasticity = True

# Run with learning
lattice.run_lattice(1000)
```

## Configuration Options

### Lattice Configuration

```python
# Set simulation parameters
lattice.set_dt(0.1)  # Time step
lattice.electrical_synapse = True  # Enable electrical synapses
lattice.chemical_synapse = False   # Enable chemical synapses
lattice.parallel = True           # Enable parallel processing
```

### STDP Configuration

```python
stdp = lixirnet.PySTDP()
stdp.a_plus = 0.01      # LTP amplitude
stdp.a_minus = 0.01     # LTD amplitude
stdp.tau_plus = 20.0    # LTP time constant
stdp.tau_minus = 20.0   # LTD time constant
stdp.dt = 0.1           # Time step
```

## Error Handling

```python
try:
    lattice.run_lattice(1000)
except lixirnet.SpikingNeuralNetworksError as e:
    print(f"Simulation error: {e}")
```

## Performance Considerations

- Use GPU interface for large-scale simulations
- Enable `parallel = True` for multi-core CPU processing
- Pre-allocate history structures for better performance
- Use appropriate `dt` values (smaller for accuracy, larger for speed)

### Utilities

#### Data Processing

```python
class Statistics:
    def __init__(self, data):
        """Create statistics calculator."""

    def mean(self): """Calculate mean."""
    def std(self): """Calculate standard deviation."""
    def variance(self): """Calculate variance."""
    def skewness(self): """Calculate skewness."""
    def kurtosis(self): """Calculate kurtosis."""

class Correlation:
    def __init__(self, x, y):
        """Create correlation analyzer."""

    def pearson(self): """Pearson correlation."""
    def spearman(self): """Spearman correlation."""
    def cross_correlation(self, max_lag): """Cross-correlation."""

class FFTProcessor:
    def __init__(self, signal, sample_rate):
        """Create FFT processor."""

    def power_spectrum(self): """Power spectrum."""
    def phase_spectrum(self): """Phase spectrum."""

class SignalProcessor:
    def __init__(self, signal):
        """Create signal processor."""

    def filter(self, filter_type, **params): """Apply filter."""
    def convolve(self, kernel): """Convolve with kernel."""
```

#### Performance Monitoring

```python
class Profiler:
    def __init__(self):
        """Create performance profiler."""

    def start_timing(self, label): """Start timing operation."""
    def stop_timing(self, label): """Stop timing operation."""
    def get_report(self): """Get performance report."""

class PerformanceReport:
    def __init__(self, profiler_data):
        """Create performance report."""

    def summary(self): """Print summary."""
    def detailed_breakdown(self): """Detailed timing breakdown."""

class MemoryTracker:
    def __init__(self):
        """Create memory usage tracker."""

    def snapshot(self): """Take memory snapshot."""
    def diff(self, snapshot1, snapshot2): """Calculate memory difference."""
```

#### Validation Functions

```python
def validate_spike_train(spike_train, max_time=None, min_neuron_id=0, max_neuron_id=None):
    """Validate spike train data.

    Args:
        spike_train (List[Spike]): Spike train to validate
        max_time (float, optional): Maximum allowed time
        min_neuron_id (int): Minimum neuron ID
        max_neuron_id (int, optional): Maximum neuron ID

    Returns:
        bool: Validation result
    """

def validate_network_parameters(num_neurons, connectivity, dt):
    """Validate network parameters.

    Args:
        num_neurons (int): Number of neurons
        connectivity (float): Connection probability
        dt (float): Time step

    Raises:
        ValueError: If parameters are invalid
    """
```

#### Data Conversion

```python
def spike_times_to_binary_array(spike_times, time_bins):
    """Convert spike times to binary array.

    Args:
        spike_times (List[float]): Spike times
        time_bins (ndarray): Time bin edges

    Returns:
        ndarray: Binary spike array
    """

def binary_array_to_spike_times(binary_array, time_bins):
    """Convert binary array to spike times.

    Args:
        binary_array (ndarray): Binary spike data
        time_bins (ndarray): Time bin centers

    Returns:
        List[float]: Spike times
    """

def interpolate_missing_values(data, method='linear'):
    """Interpolate missing values in data.

    Args:
        data (ndarray): Data with missing values
        method (str): Interpolation method

    Returns:
        ndarray: Interpolated data
    """

def generate_test_spike_data(num_neurons, duration, firing_rate=10.0):
    """Generate test spike data.

    Args:
        num_neurons (int): Number of neurons
        duration (float): Simulation duration
        firing_rate (float): Average firing rate

    Returns:
        List[Spike]: Generated spike data
    """
```

### NIR Compilation

#### `NIRCompiler`

Compiles neural network models to Neuromorphic Intermediate Representation.

```python
class NIRCompiler:
    def __init__(self, network):
        """Create NIR compiler for network.

        Args:
            network (Network): Network to compile
        """

    def compile(self, target_hardware=None):
        """Compile network to NIR.

        Args:
            target_hardware (str, optional): Target hardware platform

        Returns:
            NIRCompilationResult: Compilation result
        """

    def optimize(self, optimization_level='O2'):
        """Optimize compiled network.

        Args:
            optimization_level (str): Optimization level
        """

class NIRExecutionResult:
    def __init__(self, compilation_result):
        """NIR execution result."""

    @property
    def latency(self):
        """Execution latency."""

    @property
    def energy_consumption(self):
        """Energy consumption."""

    def get_output_spikes(self):
        """Get output spike data."""
```

### Data Formats

#### `NDFHypergraph`

Neuromorphic Data Format for hypergraph structures.

```python
class NDFHypergraph:
    def __init__(self, filename=None):
        """Create NDF hypergraph handler.

        Args:
            filename (str, optional): File to load
        """

    def load(self, filename):
        """Load from NDF file."""

    def save(self, filename):
        """Save to NDF file."""

    def to_network(self):
        """Convert to SHNN Network."""

    @classmethod
    def from_network(cls, network):
        """Create from SHNN Network."""
```

#### `DataFormatExporter`

Export data in various formats.

```python
class DataFormatExporter:
    def __init__(self, data):
        """Create data exporter.

        Args:
            data: Data to export
        """

    def to_csv(self, filename):
        """Export to CSV."""

    def to_json(self, filename):
        """Export to JSON."""

    def to_numpy(self):
        """Export to NumPy arrays."""

    def to_pandas(self):
        """Export to pandas DataFrame."""
```

#### `FormatConverter`

Convert between data formats.

```python
class FormatConverter:
    @staticmethod
    def csv_to_spikes(filename, time_col='time', neuron_col='neuron'):
        """Convert CSV to spike data."""

    @staticmethod
    def spikes_to_csv(spike_data, filename):
        """Convert spike data to CSV."""

    @staticmethod
    def numpy_to_spikes(arrays):
        """Convert NumPy arrays to spikes."""

    @staticmethod
    def spikes_to_numpy(spike_data):
        """Convert spikes to NumPy arrays."""
```

## Usage Examples

### Basic Network Simulation

```python
import shnn

# Create a network
network = shnn.Network(num_neurons=1000, connectivity=0.1, dt=0.001)

# Deploy to hardware (if available)
try:
    network.deploy_to_hardware(accelerator_id=0)
    print("Network deployed to hardware")
except:
    print("Using CPU simulation")

# Generate input spikes
input_spikes = shnn.create_poisson_spike_train(rate=50, duration=0.1)

# Process spikes
output_spikes = network.process_spikes(input_spikes)

# Analyze results
firing_rate = shnn.calculate_population_rate([output_spikes])
print(f"Average firing rate: {firing_rate.mean()} Hz")

# Visualize
shnn.plot_spike_raster(output_spikes, title="Network Output")
```

### Custom Neuron Model

```python
import shnn

# Create custom neuron parameters
params = shnn.NeuronParameters.adex(
    tau_m=20.0,      # Membrane time constant
    delta_t=2.0,     # Slope factor
    v_spike=0.0,     # Spike voltage
    tau_w=100.0      # Adaptation time constant
)

# Create neuron
neuron = shnn.AdExNeuron(neuron_id=0, params=params)

# Simulate for 1 second
dt = 0.001
spikes = []

for t in range(0, 1000):
    # Apply constant current
    input_current = 500.0  # pA
    spike = neuron.step(input_current, dt)
    if spike:
        spikes.append(spike)

print(f"Neuron fired {len(spikes)} times")
```

### STDP Learning

```python
import shnn
import numpy as np

# Create STDP rule
stdp = shnn.STDPRule(
    a_plus=0.01,      # LTP amplitude
    a_minus=0.01,     # LTD amplitude
    tau_plus=20.0,    # LTP time constant
    tau_minus=20.0    # LTD time constant
)

# Simulate spike pairs
pre_spikes = [10, 30, 50, 70]    # Presynaptic spike times
post_spikes = [15, 35, 45, 75]   # Postsynaptic spike times

# Apply STDP
weight_changes = []
current_weight = 0.5

for pre_t, post_t in zip(pre_spikes, post_spikes):
    delta_w = stdp.apply(pre_t, post_t)
    current_weight += delta_w
    current_weight = np.clip(current_weight, 0.0, 1.0)  # Clamp to [0,1]
    weight_changes.append(current_weight)

print(f"Final weight: {current_weight}")
```

### Hardware Acceleration

```python
import shnn

# Check available accelerators
cuda_devices = shnn.find_cuda_devices()
neuromorphic_devices = shnn.find_neuromorphic_devices()

print(f"CUDA devices: {len(cuda_devices)}")
print(f"Neuromorphic devices: {len(neuromorphic_devices)}")

# Benchmark accelerator
if cuda_devices:
    metrics = shnn.benchmark_accelerator(accelerator_id=0, test_duration=2.0)
    print(f"Throughput: {metrics['throughput']} spikes/sec")
    print(f"Latency: {metrics['latency']} ms")
```

### NumPy Integration

```python
import shnn
import numpy as np

# Generate spike data
spikes = shnn.generate_random_spikes(
    num_spikes=1000,
    time_range=(0.0, 1.0),
    neuron_ids=list(range(100))
)

# Convert to NumPy
neuron_ids, times, amplitudes = shnn.spikes_to_numpy(spikes)

# Create raster plot
raster = shnn.spikes_to_raster_matrix(
    spikes,
    time_bins=np.linspace(0, 1, 100),
    neuron_ids=list(range(100))
)

# Calculate cross-correlations
cc_matrix = shnn.spike_cross_correlation_matrix(
    [spikes[i::100] for i in range(100)],  # Split by neuron
    max_lag=0.05
)

# Visualize
shnn.plot_spike_raster(spikes, title="Random Spike Activity")
```

## Error Handling

SHNN provides comprehensive error handling with informative error messages:

```python
import shnn

try:
    network = shnn.Network(num_neurons=-1)  # Invalid parameter
except ValueError as e:
    print(f"Parameter error: {e}")

try:
    network.process_spikes([])  # Network not deployed
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Performance Tips

1. **Use Hardware Acceleration**: Deploy networks to GPU/neuromorphic hardware when available
2. **Batch Processing**: Use `process_batch()` for multiple simultaneous inputs
3. **NumPy Arrays**: Convert data to NumPy for efficient processing
4. **Memory Management**: Clear large data structures when no longer needed
5. **Profile Performance**: Use `Profiler` to identify bottlenecks
6. **Choose Appropriate Neuron Models**: LIF for speed, AdEx/Izhikevich for biological accuracy