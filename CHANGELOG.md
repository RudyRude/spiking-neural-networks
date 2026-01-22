# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-21

### Added
- **Digital Twin Framework**: Modular brain region simulation system
  - BrainRegion trait for composable brain modules
  - DigitalTwin orchestrator for multi-region connectivity
  - Pre-built modules: CorticalModule, HippocampalModule, LsmModule, ClassifierModule
  - Neuromodulated learning across regions
- **Advanced Classifiers and Regressors**:
  - STDP-based unsupervised classifier
  - R-STDP classifier with reward-modulated learning
  - LSM-based classification and regression
  - Comprehensive evaluation metrics and performance analysis
- **Enhanced Plasticity Systems**:
  - BCM rule with sliding threshold
  - Reward-modulated STDP (R-STDP)
  - Triplet STDP for temporal dependencies
  - Homeostatic plasticity and synaptic scaling
  - Meta-plasticity with neuromodulator sensitivity
  - Structural plasticity with pruning and formation
- **Neuromodulation and Pathology**:
  - Metabotropic neurotransmitters (dopamine, serotonin, acetylcholine)
  - Astrocyte models with tripartite synapses
  - Virtual pathology simulation (schizophrenia, Alzheimer's)
  - Virtual medication system for treatment modeling
- **Documentation Enhancements**:
  - Comprehensive advanced usage guide
  - Troubleshooting guide with common issues and solutions
  - Example-based tutorials for classifiers, digital twin, and plasticity
  - Updated API documentation with stability badges
  - Production-ready release documentation

### Changed
- **Performance Optimizations**: Increased BufWriter capacity to 8MB for I/O performance
- **API Stability**: Added stability indicators (Stable/Experimental/Unstable) across all APIs
- **Documentation**: Complete rewrite removing outdated content, adding current features

### Fixed
- **Compilation Issues**: Resolved multiple compilation errors in test suites
- **Type Annotations**: Fixed neuron pool type annotations
- **API Consistency**: Ensured consistent terminology across documentation

### Security
- **Input Validation**: Enhanced parameter validation for numerical stability
- **Memory Safety**: Rust-based implementation ensures memory safety guarantees

---

## [0.1.0] - 2024-01-20

### Added
- **Initial Release**: First stable release of HSNN framework
- **Core Architecture**: Modular spiking neural network framework in Rust
- **Neuron Models**:
  - Izhikevich neuron model with parameter sets for different behaviors
  - Morris-Lecar neuron model with calcium dynamics
  - Hodgkin-Huxley neuron model with detailed ion channels
  - Leaky Integrate-and-Fire (LIF) model
- **Network Structures**:
  - 2D lattice networks with customizable connectivity
  - Hypergraph-based connectivity for complex topologies
  - Multi-population network support
- **Synaptic Dynamics**:
  - Electrical synapses (gap junctions) - default mode
  - Chemical synapses with neurotransmitter kinetics
  - Basic neurotransmitter models (AMPA, GABA, NMDA)
  - Spike-Timing-Dependent Plasticity (STDP)
- **Python Bindings**: Complete Python interface via PyO3 (`lixirnet` package)
- **GPU Support**: CUDA and OpenCL acceleration for large-scale simulations
- **Performance Features**:
  - Parallel processing with Rayon
  - Lock-free data structures
  - Memory-efficient storage formats
- **Analysis Tools**: Built-in spike analysis, rate computation, and visualization
- **Examples**: Comprehensive example gallery including:
  - Basic raster plots
  - Head direction models
  - Grid cell simulations
  - Memory networks (Hopfield)
  - Bayesian inference pipelines
- **Documentation**: Complete documentation suite with tutorials and API reference

### Changed
- Initial release - no changes from previous versions

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

---

## Version History

### Pre-Release Versions

#### [0.0.1-alpha] - 2023-06-XX
- Alpha release with core functionality
- Basic neuron models and lattice structures
- Initial Python bindings
- Basic examples and documentation

#### [0.0.1-beta] - 2023-09-XX
- Beta release with GPU support
- STDP implementation
- Neurotransmitter dynamics
- Performance optimizations

---

## Release Notes

For detailed release notes including migration guides and breaking changes, see [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Contributing to Changelog

When contributing to this project, please:
1. Update the changelog with your changes in the "Unreleased" section
2. Follow the format: `### Added/Changed/Deprecated/Removed/Fixed/Security`
3. Keep entries concise but descriptive
4. Reference issue/PR numbers when applicable

### Types of Changes
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

---

*For the latest updates, see the [GitHub Releases](https://github.com/hsnn-project/hsnn/releases) page.*