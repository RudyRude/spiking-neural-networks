# Installation Guide

This guide provides streamlined installation instructions for the HSNN framework, prioritizing the Python interface for scientific users.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Primary Installation](#primary-installation)
- [Advanced Installation](#advanced-installation)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## Prerequisites

### Primary Requirements

1. **Python 3.8+** with pip (most important for scientific users)
2. **Git** for cloning the repository
3. **Internet connection** for downloading dependencies

### System Requirements

- **Operating System**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **RAM**: 4GB minimum, 8GB+ recommended for larger simulations
- **Disk Space**: 2GB for installation
- **Internet**: Required for initial setup

### Optional Requirements

For advanced features:
- **CUDA Toolkit 11.0+** (NVIDIA GPU acceleration)
- **OpenCL 2.0+** (GPU acceleration on AMD/Intel hardware)
- **Rust nightly toolchain** (direct Rust development)
- **Docker** (containerized deployment)

## Primary Installation

For scientific users and most applications, use the Python interface:

```bash
# 1. Clone the repository
git clone https://github.com/hsnn-project/hsnn.git
cd hsnn

# 2. Install the Python package
pip install maturin
maturin develop --release

# 3. Verify installation
python -c "import lixirnet as ln; print('✓ Installation successful!')"
```

**That's it!** The above steps install everything needed for most use cases. Rust and other dependencies are handled automatically.

## Advanced Installation

### Direct Rust Development

For contributors needing full Rust access:

```bash
# Install nightly Rust toolchain
rustup install nightly
rustup override set nightly

# Build and test
cargo build --release
cargo test
```

### GPU Acceleration

**NVIDIA GPUs (CUDA):**
```bash
# Install CUDA toolkit from NVIDIA website
# Then enable CUDA support
cd backend
cargo build --release --features cuda
```

**AMD/Intel GPUs (OpenCL):**
```bash
# Install OpenCL drivers for your hardware
# Enable OpenCL support
cd backend
cargo build --release --features opencl
```

### Docker Deployment

```bash
# Build container
docker build -t hsnn .

# Run with GPU support
docker run --gpus all -it hsnn
```

### WebAssembly & Embedded

For specialized deployments, see the [Rust API documentation](docs/api/rust-api.md).

## Platform-Specific Notes

### macOS
- Xcode Command Line Tools required: `xcode-select --install`
- Use Homebrew for easy dependency management

### Linux
- Ubuntu/Debian: `sudo apt install build-essential cmake git python3 python3-pip`
- CentOS/RHEL: `sudo dnf install gcc gcc-c++ cmake git python3 python3-pip`

### Windows
- Install Visual Studio Build Tools for C++ compilation
- Use Chocolatey for package management: `choco install python cmake git`

## Troubleshooting

### Primary Installation Issues

**"maturin command not found"**
```bash
# Ensure pip is installed and up to date
pip install --upgrade pip
pip install maturin
```

**"Python version too old"**
```bash
# Check Python version
python --version
# Must be 3.8 or higher. Update Python if needed.
```

**"Import error: No module named 'lixirnet'"**
```bash
# Re-run the installation
maturin develop --release

# Or check if you're in the correct directory
pwd  # Should be in the hsnn/ folder
```

**"Git clone fails"**
```bash
# Ensure git is installed
git --version

# Or download ZIP from GitHub and extract
```

### Advanced Installation Issues

**Rust compilation fails**
```bash
# Ensure nightly toolchain is active
rustup show

# Update Rust
rustup update nightly
```

**GPU not detected**
```bash
# For CUDA: Check installation
nvcc --version

# For OpenCL: List platforms
clinfo
```

**Build out of memory**
```bash
# Use fewer parallel jobs
cargo build --jobs 1
```

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/hsnn-project/hsnn/issues)
- **Documentation**: [Getting Started](docs/getting-started.md)
- **Community**: [Discussions](https://github.com/hsnn-project/hsnn/discussions)

## Verification

After installation, verify everything works:

### Basic Verification

```python
# Test Python interface
import lixirnet as ln

# Create a simple network
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 5, 5)
lattice.run_lattice(100)

print("✓ Basic functionality works")
```

### GPU Verification

```python
import lixirnet as ln

# Test GPU availability
if ln.gpu_available():
    print("✓ GPU support enabled")
    gpu_lattice = ln.CUDAIzhikevichLattice()
else:
    print("⚠ GPU not available")
```

### Full Test Suite

```bash
# Run Rust tests
cd backend
cargo test

# Run Python tests
python -m pytest interface/tests/

# Run benchmarks
cargo bench
```

---

For additional help, see the [Getting Started Guide](docs/getting-started.md) or visit our [GitHub repository](https://github.com/hsnn-project/hsnn).