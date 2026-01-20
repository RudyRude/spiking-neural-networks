# Development Guide

This guide provides instructions for setting up a development environment, building the project from source, running tests, and debugging.

## Prerequisites

### System Requirements

- **Rust**: Nightly toolchain (see `backend/rust-toolchain.toml`)
- **Python**: 3.8+ for Python bindings
- **Cargo**: Latest version for building Rust code
- **Maturin**: For Python package development (`pip install maturin`)
- **CUDA/OpenCL**: For GPU development (optional)

### Installing Rust

1. Install Rust using `rustup`:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Set up nightly toolchain and edition 2024:
   ```bash
   rustup install nightly
   rustup component add rust-src --toolchain nightly
   rustup override set nightly
   ```

3. Verify installation:
   ```bash
   rustc --version  # Should show nightly toolchain
   cargo --version
   ```

### Installing Python Dependencies

```bash
pip install maturin numpy scipy matplotlib
```

## Development Environment Setup

### Cloning the Repository

```bash
git clone https://github.com/your-org/spiking-neural-networks.git
cd spiking-neural-networks
```

### Setting Up Python Interface

For development builds (recommended for contributors):

```bash
cd interface
maturin develop --release
```

This installs the package in development mode with optimizations.

### GPU Development Setup

#### CUDA Setup

1. Install NVIDIA CUDA Toolkit
2. Ensure CUDA libraries are in PATH
3. For GPU lattices, use custom buffer macros: `read_and_set_buffer!`, `write_buffer!`

#### OpenCL Setup

1. Install OpenCL runtime for your platform
2. Ensure OpenCL libraries are available

## Building from Source

### Rust Backend

```bash
cd backend
cargo build --release
```

### Python Interface

```bash
cd interface
maturin build --release
pip install target/wheels/*.whl --force-reinstall
```

### GPU Components

```bash
cd gpu_impl/cuda
# Build CUDA kernels
make

cd gpu_impl/opencl
cargo build --release
```

## Testing

### Running Tests

#### Rust Tests

```bash
cd backend
cargo test
```

#### Python Tests

```bash
cd interface
python -m pytest tests/
```

#### GPU Tests

```bash
cd gpu_impl/cuda
python test_gpu.py

cd gpu_impl/opencl
cargo test
```

### Integration Tests

Run all integration tests:

```bash
# Python CPU interface tests
cd interface/tests
python -m pytest

# End-to-end experiment tests
cd interface/experiments
python run_experiments.py

# Error handling tests
cd interface
python test_error_handling.py
```

### Neural Invariants Tests

Tests validate critical neural network properties:

- `grid_formation_invariant.rs`: Ensures proper grid structure formation
- `gpu_accuracy.rs`: Validates GPU computation accuracy
- Custom neuron model tests: Verify neurotransmitter and receptor kinetics

## Debugging Guidelines

### Rust Debugging

1. Use `cargo build` with debug symbols (default)
2. Add debug prints with `println!` or `dbg!`
3. Use `gdb` or `lldb` for advanced debugging:
   ```bash
   rust-gdb target/debug/your_binary
   ```

### Python Debugging

1. Use `pdb` for interactive debugging:
   ```python
   import pdb; pdb.set_trace()
   ```

2. Run with verbose logging:
   ```bash
   python -c "import logging; logging.basicConfig(level=logging.DEBUG); your_script.py"
   ```

### GPU Debugging

1. **CUDA**: Use `cuda-memcheck` and `nvprof`
2. **OpenCL**: Use platform-specific debuggers
3. Check for kernel compilation errors
4. Validate buffer operations with custom macros

### Common Issues

- **Nightly toolchain issues**: Ensure `rust-toolchain.toml` is respected
- **Python binding issues**: Rebuild with `maturin develop --release`
- **GPU memory errors**: Check buffer sizes and synchronization
- **Trait implementation errors**: Verify `IterateAndSpike`, `NeurotransmitterKinetics`, `ReceptorKinetics` traits

## Performance Profiling

### Rust Profiling

Use `cargo flamegraph` or `perf`:

```bash
cargo install flamegraph
cargo flamegraph --bin your_binary
```

### Python Profiling

```python
import cProfile
cProfile.run('your_function()')
```

### GPU Profiling

- **CUDA**: Use `nvprof` or `nsight`
- **OpenCL**: Use platform-specific profilers

## Release Process

1. Update version numbers in `Cargo.toml` and `pyproject.toml`
2. Run full test suite
3. Update `CHANGELOG.md`
4. Create git tag
5. Build release binaries
6. Publish to PyPI and crates.io

## Additional Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Maturin Documentation](https://www.maturin.rs/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [OpenCL Specification](https://www.khronos.org/registry/OpenCL/)