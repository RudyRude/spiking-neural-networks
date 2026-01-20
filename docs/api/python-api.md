# SHNN Python API Reference

This document provides comprehensive API documentation for the Python interfaces of the SHNN (Spiking Hypergraph Neural Network) library. The Python bindings provide complete access to all SHNN functionality through PyO3, with seamless NumPy integration and hardware acceleration support.

## API Stability

SHNN follows semantic versioning (semver). API stability levels are indicated with badges:

- ![Stable](https://img.shields.io/badge/stability-stable-green) **Stable**: APIs that are mature and will maintain backward compatibility in future minor and patch releases
- ![Experimental](https://img.shields.io/badge/stability-experimental-yellow) **Experimental**: APIs that may change significantly in future releases
- ![Unstable](https://img.shields.io/badge/stability-unstable-red) **Unstable**: APIs under active development with no compatibility guarantees

## Package Overview

```python
import shnn

# Check version and features
print(f"SHNN version: {shnn.__version__}")
print(f"Available features: {shnn.FEATURES}")
```

## Core Classes

### Network Management

#### `Network` ![Stable](https://img.shields.io/badge/stability-stable-green)

The main class for creating and managing spiking neural networks.

```python
class Network:
    def __init__(self, num_neurons=1000, connectivity=0.1, dt=0.001, **kwargs):
        """Create a new spiking neural network.

        Args:
            num_neurons (int): Number of neurons in the network
            connectivity (float): Connection probability (0.0 to 1.0)
            dt (float): Time step in seconds
            **kwargs: Additional configuration options
                - input_size (int): Number of input neurons
                - output_size (int): Number of output neurons
                - hidden_layers (List[int]): Hidden layer sizes
                - topology (str): Network topology ('feedforward', 'recurrent', 'convolutional')
        """

    @classmethod
    def feedforward(cls, layer_sizes, dt=None):
        """Create a feedforward network.

        Args:
            layer_sizes (List[int]): Size of each layer
            dt (float, optional): Time step

        Returns:
            Network: Configured feedforward network
        """

    @classmethod
    def recurrent(cls, num_neurons, connectivity, dt=None):
        """Create a recurrent network.

        Args:
            num_neurons (int): Number of neurons
            connectivity (float): Connection probability
            dt (float, optional): Time step

        Returns:
            Network: Configured recurrent network
        """

    def deploy_to_hardware(self, accelerator_id=0):
        """Deploy network to hardware accelerator.

        Args:
            accelerator_id (int): ID of the accelerator to use
        """

    def process_spikes(self, input_spikes):
        """Process input spikes through the network.

        Args:
            input_spikes (List[Tuple[int, float, float]]): List of (neuron_id, time, amplitude) tuples

        Returns:
            List[Tuple[int, float, float]]: Output spikes
        """

    def process_batch(self, spike_batches):
        """Process multiple batches of spikes.

        Args:
            spike_batches (List[List[Tuple[int, float, float]]]): List of spike batches

        Returns:
            List[List[Tuple[int, float, float]]]: Processed spike batches
        """

    def update_weights(self, weight_updates):
        """Update network weights.

        Args:
            weight_updates (List[Tuple[int, int, float]]): List of (pre_neuron, post_neuron, new_weight) tuples
        """

    def get_performance_metrics(self):
        """Get network performance metrics.

        Returns:
            PerformanceMetrics: Current performance data
        """

    def reset(self):
        """Reset network to initial state."""

    def get_config(self):
        """Get network configuration as dictionary.

        Returns:
            dict: Network configuration
        """

    def update_config(self, config_dict):
        """Update network configuration.

        Args:
            config_dict (dict): New configuration values
        """

    def save_config(self, filename):
        """Save network configuration to JSON file.

        Args:
            filename (str): Output filename
        """

    @classmethod
    def load_config(cls, filename):
        """Load network configuration from JSON file.

        Args:
            filename (str): Input filename

        Returns:
            Network: Loaded network
        """

    def is_deployed(self):
        """Check if network is deployed to hardware.

        Returns:
            bool: Deployment status
        """

    def get_stats(self):
        """Get network statistics.

        Returns:
            dict: Network statistics
        """
```

#### `NetworkConfig` ![Stable](https://img.shields.io/badge/stability-stable-green)

Configuration class for network parameters.

```python
class NetworkConfig:
    def __init__(self):
        """Create a default network configuration."""

    @property
    def num_neurons(self):
        """Number of neurons."""

    @num_neurons.setter
    def num_neurons(self, value):
        """Set number of neurons."""

    @property
    def num_connections(self):
        """Number of connections."""

    @num_connections.setter
    def num_connections(self, value):
        """Set number of connections."""
```

#### `PerformanceMetrics`

Performance monitoring data.

```python
class PerformanceMetrics:
    # Performance data from hardware acceleration
    pass
```

### Neuron Models

#### `NeuronState`

Represents the internal state of a neuron.

```python
class NeuronState:
    def __init__(self, membrane_potential, recovery_variable=None):
        """Create neuron state.

        Args:
            membrane_potential (float): Membrane potential in mV
            recovery_variable (float, optional): Recovery variable
        """

    @property
    def membrane_potential(self):
        """Membrane potential (mV)."""

    @membrane_potential.setter
    def membrane_potential(self, value):
        """Set membrane potential."""

    @property
    def recovery_variable(self):
        """Recovery variable."""

    @recovery_variable.setter
    def recovery_variable(self, value):
        """Set recovery variable."""

    @property
    def last_spike_time(self):
        """Time of last spike (seconds)."""

    @property
    def adaptation_current(self):
        """Adaptation current."""

    @adaptation_current.setter
    def adaptation_current(self, value):
        """Set adaptation current."""

    @property
    def calcium_concentration(self):
        """Calcium concentration."""

    @calcium_concentration.setter
    def calcium_concentration(self, value):
        """Set calcium concentration."""

    @property
    def is_refractory(self):
        """Check if neuron is in refractory period."""

    def reset(self):
        """Reset to resting state."""
```

#### `NeuronParameters`

Configuration parameters for different neuron types.

```python
class NeuronParameters:
    def __init__(self, neuron_type="LIF", **kwargs):
        """Create neuron parameters.

        Args:
            neuron_type (str): Type of neuron ('LIF', 'AdEx', 'Izhikevich')
            **kwargs: Neuron-specific parameters
        """

    @classmethod
    def lif(cls, tau_m=None, v_threshold=None, v_reset=None):
        """Create LIF neuron parameters.

        Args:
            tau_m (float, optional): Membrane time constant
            v_threshold (float, optional): Threshold voltage
            v_reset (float, optional): Reset voltage
        """

    @classmethod
    def adex(cls, tau_m=None, delta_t=None, v_spike=None, tau_w=None):
        """Create AdEx neuron parameters.

        Args:
            tau_m (float, optional): Membrane time constant
            delta_t (float, optional): Slope factor
            v_spike (float, optional): Spike voltage
            tau_w (float, optional): Adaptation time constant
        """

    @classmethod
    def izhikevich(cls, a=None, b=None, c=None, d=None):
        """Create Izhikevich neuron parameters.

        Args:
            a (float, optional): Recovery time constant
            b (float, optional): Recovery sensitivity
            c (float, optional): Reset voltage
            d (float, optional): Reset recovery
        """

    @classmethod
    def regular_spiking(cls):
        """Create regular spiking neuron parameters."""

    @classmethod
    def fast_spiking(cls):
        """Create fast spiking neuron parameters."""

    @classmethod
    def chattering(cls):
        """Create chattering neuron parameters."""

    @classmethod
    def bursting(cls):
        """Create bursting neuron parameters."""

    # Parameter properties
    @property
    def tau_m(self): """Membrane time constant."""
    @property
    def v_threshold(self): """Threshold voltage."""
    @property
    def v_reset(self): """Reset voltage."""
    @property
    def refractory_period(self): """Refractory period."""
```

#### Individual Neuron Classes

```python
class LIFNeuron:
    def __init__(self, neuron_id, params=None):
        """Create a Leaky Integrate-and-Fire neuron.

        Args:
            neuron_id (int): Unique neuron identifier
            params (NeuronParameters, optional): Neuron parameters
        """

    def step(self, input_current, dt):
        """Advance neuron by one time step.

        Args:
            input_current (float): Input current
            dt (float): Time step

        Returns:
            Spike or None: Spike if fired, None otherwise
        """

    def reset(self):
        """Reset neuron to resting state."""

    @property
    def voltage(self):
        """Current membrane potential."""

class AdExNeuron:
    def __init__(self, neuron_id, params=None):
        """Create an Adaptive Exponential neuron.

        Args:
            neuron_id (int): Unique neuron identifier
            params (NeuronParameters, optional): Neuron parameters
        """

    # Same methods as LIFNeuron

class IzhikevichNeuron:
    def __init__(self, neuron_id, params=None):
        """Create an Izhikevich neuron.

        Args:
            neuron_id (int): Unique neuron identifier
            params (NeuronParameters, optional): Neuron parameters
        """

    # Same methods as LIFNeuron
```

#### `Spike`

Represents a spike event.

```python
class Spike:
    def __init__(self, neuron_id, time, amplitude=1.0):
        """Create a spike event.

        Args:
            neuron_id (int): ID of spiking neuron
            time (float): Spike time in seconds
            amplitude (float, optional): Spike amplitude
        """

    @property
    def neuron_id(self):
        """Neuron that spiked."""

    @property
    def time(self):
        """Spike time."""

    @property
    def amplitude(self):
        """Spike amplitude."""
```

### Spike Processing

#### `SpikeBuffer`

Manages collections of spikes.

```python
class SpikeBuffer:
    def __init__(self, capacity=None):
        """Create a spike buffer.

        Args:
            capacity (int, optional): Buffer capacity
        """

    def add_spike(self, spike):
        """Add a spike to the buffer."""

    def get_spikes_in_range(self, start_time, end_time):
        """Get spikes within time range."""

    def clear(self):
        """Clear all spikes."""
```

#### `SpikePattern`

Analyzes spike patterns and timing.

```python
class SpikePattern:
    def __init__(self, spikes):
        """Create spike pattern analyzer.

        Args:
            spikes (List[Spike]): Spike data
        """

    def calculate_isi(self):
        """Calculate inter-spike intervals."""

    def calculate_firing_rate(self):
        """Calculate firing rate."""

    def detect_bursts(self):
        """Detect burst patterns."""
```

#### Spike Generators

```python
# Generate Poisson spike train
def create_poisson_spike_train(rate, duration, neuron_id=0):
    """Generate Poisson spike train.

    Args:
        rate (float): Firing rate (Hz)
        duration (float): Duration (seconds)
        neuron_id (int, optional): Neuron ID

    Returns:
        List[Spike]: Generated spikes
    """

# Generate regular spike train
def create_regular_spike_train(interval, duration, neuron_id=0):
    """Generate regular spike train.

    Args:
        interval (float): Inter-spike interval (seconds)
        duration (float): Duration (seconds)
        neuron_id (int, optional): Neuron ID

    Returns:
        List[Spike]: Generated spikes
    """

# Generate burst spike train
def create_burst_spike_train(burst_size, burst_interval, inter_burst_interval, duration, neuron_id=0):
    """Generate bursting spike train.

    Args:
        burst_size (int): Spikes per burst
        burst_interval (float): Inter-spike interval within burst
        inter_burst_interval (float): Interval between bursts
        duration (float): Total duration
        neuron_id (int, optional): Neuron ID

    Returns:
        List[Spike]: Generated spikes
    """

def generate_random_spikes(num_spikes, time_range, neuron_ids):
    """Generate random spikes.

    Args:
        num_spikes (int): Number of spikes to generate
        time_range (Tuple[float, float]): Time range (start, end)
        neuron_ids (List[int]): Available neuron IDs

    Returns:
        List[Spike]: Random spikes
    """
```

#### Spike Encoders

```python
class PoissonEncoder:
    def __init__(self, rate_range=(0.0, 100.0)):
        """Create Poisson rate encoder.

        Args:
            rate_range (Tuple[float, float]): Min/max firing rates
        """

    def encode(self, values):
        """Encode values as spike trains."""

class TemporalEncoder:
    def __init__(self, time_window):
        """Create temporal encoder.

        Args:
            time_window (float): Encoding time window
        """

    def encode(self, values):
        """Encode values with temporal coding."""

class RateEncoder:
    def __init__(self, threshold):
        """Create rate encoder.

        Args:
            threshold (float): Encoding threshold
        """

    def encode(self, values):
        """Encode values as firing rates."""
```

#### Spike Analysis Functions

```python
def calculate_spike_train_distance(train1, train2, method="victor_purpura"):
    """Calculate distance between spike trains.

    Args:
        train1 (List[Spike]): First spike train
        train2 (List[Spike]): Second spike train
        method (str): Distance metric

    Returns:
        float: Distance measure
    """

def detect_bursts(spike_train, min_burst_size=3, max_isi=0.01):
    """Detect bursts in spike train.

    Args:
        spike_train (List[Spike]): Input spikes
        min_burst_size (int): Minimum spikes per burst
        max_isi (float): Maximum inter-spike interval for burst

    Returns:
        List[List[Spike]]: Detected bursts
    """

def calculate_population_rate(spike_trains, bin_size=0.01):
    """Calculate population firing rate.

    Args:
        spike_trains (List[List[Spike]]): Multiple spike trains
        bin_size (float): Time bin size

    Returns:
        ndarray: Population rate over time
    """

def calculate_synchrony_index(spike_trains, time_window):
    """Calculate spike synchrony index.

    Args:
        spike_trains (List[List[Spike]]): Spike trains to analyze
        time_window (float): Analysis window

    Returns:
        float: Synchrony index
    """
```

### Plasticity and Learning

#### Plasticity Rules

```python
class STDPRule:
    def __init__(self, a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0,
                 rule_type="additive", w_min=0.0, w_max=1.0, tau_y=None):
        """Create STDP learning rule.

        Args:
            a_plus (float): LTP amplitude
            a_minus (float): LTD amplitude
            tau_plus (float): LTP time constant
            tau_minus (float): LTD time constant
            rule_type (str): STDP rule type
            w_min (float): Minimum weight
            w_max (float): Maximum weight
            tau_y (float, optional): Additional time constant
        """

    def apply(self, pre_time, post_time):
        """Apply STDP weight update.

        Args:
            pre_time (float): Presynaptic spike time
            post_time (float): Postsynaptic spike time

        Returns:
            float: Weight change
        """

class HomeostaticRule:
    def __init__(self, target_rate, learning_rate):
        """Create homeostatic plasticity rule.

        Args:
            target_rate (float): Target firing rate
            learning_rate (float): Learning rate
        """

class BCMRule:
    def __init__(self, learning_rate, theta_tau):
        """Create BCM learning rule.

        Args:
            learning_rate (float): Learning rate
            theta_tau (float): Threshold time constant
        """

class OjaRule:
    def __init__(self, learning_rate, beta):
        """Create Oja's learning rule.

        Args:
            learning_rate (float): Learning rate
            beta (float): Forgetting factor
        """

class HebbianRule:
    def __init__(self, learning_rate):
        """Create Hebbian learning rule.

        Args:
            learning_rate (float): Learning rate
        """
```

#### Plasticity Analysis

```python
def plot_stdp_window(rule, time_range=(-50, 50)):
    """Plot STDP learning window.

    Args:
        rule (STDPRule): STDP rule to plot
        time_range (Tuple[float, float]): Time range to plot
    """

def simulate_weight_evolution(rule, spike_pairs, initial_weight=0.5):
    """Simulate weight evolution over time.

    Args:
        rule: Plasticity rule
        spike_pairs (List[Tuple[float, float]]): Pre/post spike time pairs
        initial_weight (float): Starting weight

    Returns:
        List[float]: Weight evolution
    """

def calculate_weight_distribution(weights, bins=50):
    """Calculate weight distribution statistics.

    Args:
        weights (List[float]): Synaptic weights
        bins (int): Number of histogram bins

    Returns:
        dict: Distribution statistics
    """
```

### Hardware Acceleration

#### `AcceleratorRegistry`

Manages hardware accelerators.

```python
class AcceleratorRegistry:
    @staticmethod
    def get_available_accelerators():
        """Get list of available accelerators."""

    @staticmethod
    def find_cuda_devices():
        """Find CUDA-compatible devices.

        Returns:
            List[dict]: CUDA device information
        """

    @staticmethod
    def find_neuromorphic_devices():
        """Find neuromorphic hardware.

        Returns:
            List[dict]: Neuromorphic device information
        """

    @staticmethod
    def benchmark_accelerator(accelerator_id, test_duration=1.0):
        """Benchmark accelerator performance.

        Args:
            accelerator_id (int): Accelerator to benchmark
            test_duration (float): Benchmark duration

        Returns:
            dict: Performance metrics
        """
```

#### Accelerator Types

```python
class AcceleratorType:
    @property
    def name(self):
        """Accelerator name."""

    def is_neuromorphic(self):
        """Check if neuromorphic hardware."""

    def is_parallel(self):
        """Check if parallel processing capable."""

class AcceleratorCapabilities:
    # Hardware capabilities
    pass

class AcceleratorInfo:
    # Detailed accelerator information
    pass

class HardwareStatus:
    # Current hardware status
    pass
```

### Connectivity

#### Network Connectivity Classes

```python
class HypergraphNetwork:
    def __init__(self):
        """Create hypergraph-based connectivity."""

    def add_hyperedge(self, sources, targets):
        """Add hyperedge connection."""

    def get_hyperedges(self, neuron_id):
        """Get hyperedges for neuron."""

class GraphNetwork:
    def __init__(self):
        """Create traditional graph connectivity."""

class MatrixNetwork:
    def __init__(self, num_neurons):
        """Create dense matrix connectivity."""

class SparseMatrixNetwork:
    def __init__(self, num_neurons):
        """Create sparse matrix connectivity."""

class PlasticConnectivity:
    def __init__(self, base_connectivity):
        """Create plastic connectivity layer."""

    def update_weights(self, updates):
        """Update synaptic weights."""
```

### Visualization

#### Plotting Classes

```python
class RasterPlot:
    def __init__(self, spike_data):
        """Create raster plot.

        Args:
            spike_data: Spike data to plot
        """

    def show(self):
        """Display the plot."""

class MembraneTrace:
    def __init__(self, voltage_data, time_data):
        """Create membrane potential trace.

        Args:
            voltage_data: Voltage values over time
            time_data: Time values
        """

class WeightMatrix:
    def __init__(self, weights):
        """Create weight matrix visualization.

        Args:
            weights: Synaptic weight matrix
        """
```

#### Visualization Functions

```python
def create_raster_plot(spike_data, **kwargs):
    """Create and return raster plot.

    Args:
        spike_data: Spike events to plot
        **kwargs: Plotting options

    Returns:
        RasterPlot: Configured plot
    """

def create_membrane_trace(voltage_data, time_data, **kwargs):
    """Create membrane potential trace.

    Args:
        voltage_data: Voltage values
        time_data: Time values
        **kwargs: Plotting options

    Returns:
        MembraneTrace: Configured trace
    """

def create_weight_matrix(weights, **kwargs):
    """Create weight matrix visualization.

    Args:
        weights: Weight matrix
        **kwargs: Plotting options

    Returns:
        WeightMatrix: Configured visualization
    """

# Convenience plotting functions
def plot_spike_raster(spike_data, title=None, **kwargs):
    """Plot spike raster directly."""

def plot_membrane_potential(voltage_data, time_data, title=None, **kwargs):
    """Plot membrane potential directly."""

def plot_weight_matrix(weights, title=None, **kwargs):
    """Plot weight matrix directly."""

def plot_firing_rate(spike_data, bin_size=0.01, title=None, **kwargs):
    """Plot firing rate histogram."""
```

### NumPy Integration

#### Array Conversion Functions

```python
def spikes_to_numpy(spike_list):
    """Convert spike list to NumPy arrays.

    Args:
        spike_list (List[Spike]): Spike events

    Returns:
        Tuple[ndarray, ndarray, ndarray]: (neuron_ids, times, amplitudes)
    """

def numpy_to_spikes(neuron_ids, times, amplitudes=None):
    """Convert NumPy arrays to spike list.

    Args:
        neuron_ids (ndarray): Neuron IDs
        times (ndarray): Spike times
        amplitudes (ndarray, optional): Spike amplitudes

    Returns:
        List[Spike]: Spike events
    """

def spikes_to_raster_matrix(spike_list, time_bins, neuron_ids=None):
    """Convert spikes to raster matrix.

    Args:
        spike_list (List[Spike]): Spike events
        time_bins (ndarray): Time bin edges
        neuron_ids (List[int], optional): Neuron IDs to include

    Returns:
        ndarray: Binary raster matrix
    """

def spikes_to_density_matrix(spike_list, time_bins, neuron_ids, sigma=1.0):
    """Convert spikes to density matrix.

    Args:
        spike_list (List[Spike]): Spike events
        time_bins (ndarray): Time bin edges
        neuron_ids (List[int]): Neuron IDs
        sigma (float): Gaussian smoothing sigma

    Returns:
        ndarray: Smoothed density matrix
    """

def calculate_population_vector(spike_trains, time_window):
    """Calculate population vector from spike trains.

    Args:
        spike_trains (List[List[Spike]]): Multiple spike trains
        time_window (float): Analysis window

    Returns:
        ndarray: Population vectors
    """

def spike_triggered_average(signal, spike_times, window=(-0.1, 0.1)):
    """Calculate spike-triggered average.

    Args:
        signal (ndarray): Continuous signal
        spike_times (ndarray): Spike times
        window (Tuple[float, float]): Analysis window

    Returns:
        Tuple[ndarray, ndarray]: (average, time_axis)
    """

def weights_to_numpy(weight_matrix):
    """Convert weight matrix to NumPy array."""

def numpy_to_weights(numpy_array):
    """Convert NumPy array to weight matrix."""

def spike_cross_correlation_matrix(spike_trains, max_lag, bin_size=0.001):
    """Calculate cross-correlation matrix.

    Args:
        spike_trains (List[List[Spike]]): Spike trains
        max_lag (float): Maximum lag time
        bin_size (float): Time bin size

    Returns:
        ndarray: Cross-correlation matrix
    """

def validate_numpy_array(array, dtype=None, shape=None):
    """Validate NumPy array properties.

    Args:
        array (ndarray): Array to validate
        dtype: Expected data type
        shape: Expected shape

    Returns:
        bool: Validation result
    """
```

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