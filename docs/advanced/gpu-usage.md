# Advanced Tutorial: GPU Acceleration

This tutorial covers GPU acceleration for spiking neural network simulations using OpenCL, enabling significant performance improvements for large-scale models.

## Introduction to GPU Acceleration

GPU acceleration leverages parallel processing capabilities of modern GPUs to simulate thousands of neurons simultaneously. The framework currently supports OpenCL (cross-platform) backend, with CUDA support planned for future releases.

> **Note**: CUDA support is not yet implemented. The framework uses OpenCL for GPU acceleration on NVIDIA, AMD, and Intel GPUs.

## Hardware Requirements

### OpenCL Setup (Cross-Platform)

OpenCL is typically pre-installed on most systems. For AMD GPUs on Linux:

```bash
# Install AMDGPU drivers (Ubuntu)
sudo apt install mesa-opencl-icd

# Verify installation
clinfo
```

On macOS and Windows, OpenCL drivers are usually included with GPU drivers.

## Basic GPU Lattice Creation

```python
import lixirnet as ln

# Create GPU-accelerated lattice (OpenCL backend)
gpu_lattice = ln.IzhikevichNeuronLatticeGPU()
gpu_lattice.populate(ln.IzhikevichNeuron(), 100, 100)  # 10,000 neurons

# The lattice automatically selects the best available GPU device
print(f"GPU Device: Available via OpenCL")
print(f"Backend: OpenCL")
```

## GPU-Specific Connectivity

### Efficient Sparse Connectivity

GPU performance benefits greatly from sparse connectivity patterns.

```python
import numpy as np

# Create sparse random connectivity (1% connection probability)
def sparse_connection(pos1, pos2):
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance <= 5 and np.random.rand() < 0.01

# GPU-optimized connection with compressed sparse row (CSR) format
gpu_lattice.connect_sparse(
    connection_func=sparse_connection,
    format='csr',              # Compressed Sparse Row
    max_connections=500        # Maximum synapses per neuron
)

# Verify sparsity
print(f"Connection matrix sparsity: {gpu_lattice.get_sparsity():.4f}")
print(f"Total synapses: {gpu_lattice.get_synapse_count()}")
```

### Block-Structured Connectivity

For optimal GPU memory access patterns:

```python
# Block connectivity (neurons connect within blocks)
def block_connection(pos1, pos2, block_size=10):
    block1_x, block1_y = pos1[0] // block_size, pos1[1] // block_size
    block2_x, block2_y = pos2[0] // block_size, pos2[1] // block_size

    # Connect within same block or adjacent blocks
    return (abs(block1_x - block2_x) <= 1 and
            abs(block1_y - block2_y) <= 1 and
            pos1 != pos2)

gpu_lattice.connect_blockwise(
    block_connection,
    block_size=10,
    use_texture_memory=True  # Optimize for GPU texture cache
)
```

## GPU Memory Management

### Buffer Allocation

```python
# Custom buffer allocation for large networks
buffer_config = ln.GPUBufferConfig(
    neuron_states='device',      # Store neuron states on GPU
    synaptic_weights='device',   # Store weights on GPU
    spike_queues='pinned',       # Pinned host memory for spike queues
    history_buffers='zero_copy'  # Zero-copy memory for history
)

gpu_lattice.configure_buffers(buffer_config)

# Monitor memory usage
memory_info = gpu_lattice.get_memory_usage()
print(f"GPU Memory Used: {memory_info['used'] / 1024**2:.1f} MB")
print(f"GPU Memory Free: {memory_info['free'] / 1024**2:.1f} MB")
```

### Memory Transfer Optimization

```python
# Asynchronous memory transfers
gpu_lattice.enable_async_transfers()

# Prefetch data to GPU
gpu_lattice.prefetch_to_device()

# Overlap computation with data transfer
with gpu_lattice.async_context():
    # Computation happens here
    gpu_lattice.run_lattice_async(1000)

    # Data transfer happens simultaneously
    weights = gpu_lattice.get_weights_async()
```

## Running GPU Simulations

### Basic GPU Simulation

```python
# Setup GPU lattice
gpu_lattice = ln.IzhikevichNeuronLatticeGPU()
gpu_lattice.populate(ln.IzhikevichNeuron(), 50, 50)
gpu_lattice.connect(sparse_connection)

# Configure for performance
gpu_lattice.set_block_size(256)  # OpenCL workgroup size
gpu_lattice.enable_fast_math()   # Use fast math approximations

# Warm up GPU
gpu_lattice.warm_up()

# Run simulation
start_time = ln.gpu_time()
gpu_lattice.run_lattice_gpu(10000)
end_time = ln.gpu_time()

print(f"GPU simulation time: {end_time - start_time:.3f} seconds")
print(f"Performance: {gpu_lattice.size * 10000 / (end_time - start_time):.0f} neuron-updates/second")
```

### Multi-GPU Simulations

> **Note**: Multi-GPU support is not yet implemented. Currently, only single GPU acceleration is supported.

```python
# Future multi-GPU setup (not yet available)
# gpu_manager = ln.MultiGPUManager()
# distributed_lattice = gpu_manager.create_distributed_lattice(...)
```

## GPU-Accelerated Plasticity

> **Note**: GPU-accelerated plasticity is not yet implemented. Plasticity rules currently run on CPU.

```python
# CPU plasticity (GPU support planned)
stdp = ln.STDPRule(
    a_plus=0.01,
    a_minus=0.0105,
    tau_plus=20.0,
    tau_minus=20.0
)

gpu_lattice.connect(
    sparse_connection,
    plasticity_rule=stdp
)

# Plasticity simulation (runs on CPU)
for epoch in range(100):
    gpu_lattice.run_plasticity_epoch(1000)
    weight_stats = gpu_lattice.get_weight_statistics()
    print(f"Epoch {epoch}: Mean weight = {weight_stats['mean']:.4f}")
```

## Advanced GPU Features

> **Note**: Advanced features like custom kernels, detailed timing, and memory pools are not yet implemented.

```python
# Basic GPU timing (available)
import time
start = time.time()
gpu_lattice.run_lattice_gpu(1000)
end = time.time()
print(f"GPU time: {end - start:.3f} s")
```

## Performance Optimization

> **Note**: Advanced optimization features are not yet implemented. Basic performance tuning is available.

```python
# Basic performance configuration
gpu_lattice.set_block_size(256)  # Workgroup size
gpu_lattice.enable_fast_math()   # Use approximations for speed

# Benchmark basic performance
time_taken = gpu_lattice.benchmark(1000)
print(f"Time for 1000 iterations: {time_taken:.3f} s")
```

## Debugging GPU Code

> **Note**: Advanced debugging and profiling tools are not yet implemented. Basic error handling is available.

```python
# Basic GPU simulation with error handling
try:
    gpu_lattice.run_lattice_gpu(1000)
    print("Simulation completed successfully")
except Exception as e:
    print(f"GPU Error: {e}")
    # Check OpenCL installation and GPU drivers
```

## Multi-Platform Support

### Automatic Device Selection

```python
# GPU lattice automatically selects best available GPU
gpu_lattice = ln.IzhikevichNeuronLatticeGPU()
gpu_lattice.populate(ln.IzhikevichNeuron(), 100, 100)

print("Backend: OpenCL")
print("Device: Automatically selected GPU")
```

### OpenCL Optimizations

```python
# Basic OpenCL optimizations
gpu_lattice.set_block_size(256)  # Workgroup size
gpu_lattice.enable_fast_math()   # Fast math approximations
```

## Performance Benchmarks

### Scaling Performance

```python
sizes = [10, 25, 50, 100, 200]

for size in sizes:
    lattice = ln.CUDAIzhikevichLattice()
    lattice.populate(ln.IzhikevichNeuron(), size, size)
    lattice.connect(lambda x, y: np.random.rand() < 0.01)

    # Benchmark
    time = lattice.benchmark(1000)
    neurons = size * size

    print(f"{neurons:5d} neurons: {time:6.3f} s "
          f"({neurons/time:8.0f} neurons/s)")
```

### GPU vs CPU Comparison

```python
# CPU version
cpu_lattice = ln.IzhikevichLattice()
cpu_lattice.populate(ln.IzhikevichNeuron(), 50, 50)
cpu_time = cpu_lattice.benchmark(1000)

# GPU version
gpu_lattice = ln.IzhikevichNeuronLatticeGPU()
gpu_lattice.populate(ln.IzhikevichNeuron(), 50, 50)
gpu_time = gpu_lattice.benchmark(1000)

print(f"CPU time: {cpu_time:.3f} s")
print(f"GPU time: {gpu_time:.3f} s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

## Current Limitations

The GPU acceleration implementation is in active development. Current limitations include:

- **Backend**: Only OpenCL is supported (CUDA planned for future)
- **Features**: 
  - Chemical synapses not yet supported on GPU
  - Plasticity runs on CPU (GPU support planned)
  - Multi-GPU not supported
  - Advanced profiling/debugging tools not available
- **Compatibility**: Requires OpenCL 1.2+ compatible GPU and drivers
- **Performance**: Optimal for electrical synapse simulations with large lattices

## Troubleshooting

### Common GPU Issues

- **OpenCL Platform Errors**: Verify OpenCL drivers are installed (`clinfo` command)
- **Device Selection Errors**: Ensure a compatible GPU is available
- **Memory Issues**: Monitor GPU memory usage, reduce lattice size if needed
- **Performance Issues**: Adjust workgroup size, check OpenCL kernel compilation

### Memory Issues

```python
# Check available memory
if gpu_lattice.get_available_memory() < required_memory:
    # Reduce lattice size or disable history
    gpu_lattice.disable_history()
```

### Kernel Launch Failures

```python
# Diagnose kernel failures
try:
    cuda_lattice.run_lattice_gpu(1000)
except ln.KernelLaunchError:
    # Check kernel parameters
    config = cuda_lattice.validate_kernel_config()
    if not config['valid']:
        print("Invalid kernel configuration:")
        for issue in config['issues']:
            print(f"  - {issue}")
```

## Next Steps

- Learn about [Custom Neuron Models](custom-neuron-models.md)
- Explore neurotransmitter dynamics with GPU acceleration
- See GPU examples in the [Examples Gallery](../examples/gallery.md)