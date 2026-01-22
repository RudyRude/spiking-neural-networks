# Troubleshooting Guide

This guide helps resolve common issues encountered when using HSNN. If you can't find a solution here, please check the [GitHub issues](https://github.com/hsnn-project/hsnn/issues) or create a new issue.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Compilation Errors](#compilation-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [GPU Acceleration Problems](#gpu-acceleration-problems)
- [Python Interface Issues](#python-interface-issues)

## Installation Issues

### Maturin Build Fails

**Problem**: `maturin develop --release` fails with compilation errors.

**Solutions**:

1. **Check Rust Version**:
   ```bash
   rustc --version  # Should be 1.70+
   cargo --version  # Should be compatible
   ```

2. **Update Dependencies**:
   ```bash
   pip install --upgrade maturin numpy scipy
   ```

3. **Clean Build**:
   ```bash
   rm -rf backend/target/
   maturin develop --release
   ```

### Python Import Error

**Problem**: `import lixirnet as ln` fails.

**Solutions**:

1. **Reinstall Package**:
   ```bash
   pip uninstall lixirnet
   maturin develop --release
   ```

2. **Check Python Path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Virtual Environment Issues**:
   ```bash
   # Create new virtual environment
   python -m venv hsnn_env
   source hsnn_env/bin/activate  # Linux/Mac
   pip install maturin numpy
   maturin develop --release
   ```

## Compilation Errors

### Nightly Rust Toolchain Issues

**Problem**: Compilation fails with unstable feature errors.

**Solutions**:

1. **Set Correct Toolchain**:
   ```bash
   rustup install nightly-2024-01-01  # Specific nightly version
   rustup override set nightly-2024-01-01
   ```

2. **Update rust-toolchain.toml**:
   ```toml
   [toolchain]
   channel = "nightly-2024-01-01"
   components = ["rustfmt", "clippy"]
   targets = ["x86_64-unknown-linux-gnu"]
   ```

### Missing Dependencies

**Problem**: Linker errors for system libraries.

**Solutions**:

1. **Install System Dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get install build-essential libssl-dev pkg-config
   ```

2. **macOS**:
   ```bash
   xcode-select --install
   brew install openssl
   ```

3. **Windows**:
   ```bash
   # Install Visual Studio Build Tools
   # Or use MSYS2 with mingw-w64
   ```

## Runtime Errors

### Memory Allocation Failures

**Problem**: `Out of memory` errors during large simulations.

**Solutions**:

1. **Reduce Network Size**:
   ```python
   # Use smaller networks for testing
   network = nn.Network(num_neurons=1000)  # Instead of 10000
   ```

2. **Enable Memory Pool**:
   ```python
   config = nn.SimulationConfig(
       enable_memory_pool=True,
       max_memory_mb=4096
   )
   ```

3. **Monitor Memory Usage**:
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

### Spike Overflow

**Problem**: Too many spikes generated, causing buffer overflow.

**Solutions**:

1. **Increase Spike Buffer**:
   ```python
   network = nn.Network(
       num_neurons=5000,
       spike_buffer_size=500000  # Increase buffer
   )
   ```

2. **Reduce Simulation Time**:
   ```python
   results = network.simulate(duration=1.0)  # Shorter simulation
   ```

### NaN Values in Results

**Problem**: Simulation produces NaN values.

**Solutions**:

1. **Check Parameters**:
   ```python
   # Validate neuron parameters
   assert network.parameters_valid()
   ```

2. **Reduce Time Step**:
   ```python
   results = network.simulate(dt=0.0001)  # Smaller dt
   ```

3. **Check Input Ranges**:
   ```python
   # Ensure inputs are within reasonable ranges
   inputs = np.clip(inputs, -10.0, 10.0)
   ```

## Performance Issues

### Slow Simulations

**Problem**: Simulations run slower than expected.

**Solutions**:

1. **Enable Parallel Processing**:
   ```python
   config = nn.SimulationConfig(parallel=True, num_threads=8)
   results = network.simulate_with_config(duration=10.0, config=config)
   ```

2. **Optimize Network Structure**:
   ```python
   # Use sparse connectivity
   network.set_connectivity(nn.SparseConnectivity(density=0.1))
   ```

3. **Profile Bottlenecks**:
   ```python
   profiler = nn.Profiler()
   profiler.start()
   results = network.simulate(duration=1.0)
   report = profiler.report()
   print(report)  # Identify slow components
   ```

### High CPU Usage

**Problem**: Simulation uses too much CPU.

**Solutions**:

1. **Adjust Thread Count**:
   ```python
   # Match thread count to CPU cores
   import multiprocessing
   num_cores = multiprocessing.cpu_count()
   config = nn.SimulationConfig(num_threads=num_cores)
   ```

2. **Use GPU Acceleration** (if available):
   ```python
   accelerator = nn.CUDAAccelerator()
   network.deploy_to_accelerator(accelerator)
   ```

## GPU Acceleration Problems

### CUDA Not Detected

**Problem**: GPU not found despite CUDA installation.

**Solutions**:

1. **Check CUDA Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Set Library Path** (Linux):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Rebuild with CUDA**:
   ```bash
   cargo clean
   cargo build --release --features cuda
   ```

### OpenCL Issues

**Problem**: OpenCL devices not found.

**Solutions**:

1. **Install OpenCL Drivers**:
   - AMD: Install ROCm
   - Intel: Install Intel OpenCL SDK
   - NVIDIA: CUDA includes OpenCL

2. **Check OpenCL Platforms**:
   ```python
   import pyopencl as cl
   platforms = cl.get_platforms()
   print(f"Found {len(platforms)} platforms")
   ```

### GPU Memory Errors

**Problem**: GPU out of memory.

**Solutions**:

1. **Reduce Batch Size**:
   ```python
   config = nn.GPUConfig(batch_size=512)  # Smaller batches
   ```

2. **Use CPU Fallback**:
   ```python
   network.set_accelerator(nn.CPUAccelerator())  # Fallback to CPU
   ```

## Python Interface Issues

### Type Conversion Errors

**Problem**: Errors when passing data between Python and Rust.

**Solutions**:

1. **Check Data Types**:
   ```python
   # Ensure numpy arrays are correct dtype
   spikes = np.array(spikes, dtype=np.float64)
   ```

2. **Use Correct Formats**:
   ```python
   # Convert to expected format
   input_data = nn.numpy_to_spikes(neuron_ids, times, amplitudes)
   ```

### Callback Function Issues

**Problem**: Python callback functions not working in Rust code.

**Solutions**:

1. **Use Simple Functions**:
   ```python
   def reward_fn(state):
       return state['accuracy'] * 0.1

   # Avoid complex closures
   network.set_reward_function(reward_fn)
   ```

2. **Check Function Signature**:
   ```python
   # Ensure correct parameter types
   def plasticity_callback(pre_time, post_time):
       return 0.01 * (post_time - pre_time)
   ```

### Serialization Issues

**Problem**: Errors when saving/loading network state.

**Solutions**:

1. **Check File Permissions**:
   ```bash
   touch /path/to/save/file.pkl
   ```

2. **Use Correct Format**:
   ```python
   # Use supported formats
   network.save('network.h5')  # HDF5 format
   ```

## Common Error Messages

### "Invalid neuron configuration"

**Cause**: Neuron parameters out of valid range.

**Fix**:
```python
# Check and clamp parameters
params = network.get_parameters()
params['threshold'] = np.clip(params['threshold'], 0.1, 1.0)
network.update_parameters(params)
```

### "Connectivity matrix too large"

**Cause**: Dense connectivity with many neurons.

**Fix**:
```python
# Use sparse connectivity
network.set_connectivity(nn.SparseConnectivity(density=0.05))
```

### "Time step too large"

**Cause**: Numerical instability with large dt.

**Fix**:
```python
results = network.simulate(dt=0.0001)  # Smaller time step
```

## Getting Help

### Debug Information

```python
# Get system information
info = nn.get_system_info()
print(info)

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reporting Issues

When reporting bugs, please include:

1. **HSNN Version**: `print(shnn.__version__)`
2. **Python Version**: `python --version`
3. **Rust Version**: `rustc --version`
4. **Operating System**: `uname -a`
5. **Error Traceback**: Full error message
6. **Minimal Reproducible Example**: Small code that reproduces the issue

### Community Support

- **GitHub Issues**: https://github.com/hsnn-project/hsnn/issues
- **Discussions**: https://github.com/hsnn-project/hsnn/discussions
- **Documentation**: https://hsnn-project.github.io/