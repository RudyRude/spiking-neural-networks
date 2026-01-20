# Spiking Neural Networks Project Roadmap

This document outlines the comprehensive plan to integrate and resolve all TODO items from the project documentation (`docs/spiking-neural-networks.md`). The roadmap is organized by phases, priorities, and categories, with estimated effort and dependencies.

## Overview
The project aims to build a generalized SNN system with biological accuracy, from individual neurons to cognitive simulations. Key components include neuron models, plasticity, neuromodulation, attractors, LSMs, and applications.

## Categories of TODOs

### 1. Documentation & Education
- Write explanations for neuron models (LIF, Izhikevich, HH)
- Create diagrams for lattices, attractors, Hopfield networks
- Document neurotransmission adaptations and kinetics rationale
- Sources: Izhikevich, Destexhe, biological papers

### 2. Performance & Optimization
- Increase BufWriter capacity (4-8 MB) for faster I/O
- Optimize lattice simulations (reduce dt, use Rayon/WGPU)
- Pre-allocate memory, minimize allocations
- CUDA/GPU support for electrical synapses

### 3. Core Neuron Models & Implementation
- Integrate-and-fire variants (Basic, Adaptive, Adaptive Exponential)
- Izhikevich models (regular, leaky hybrid)
- Hodgkin-Huxley with gating, neurotransmission, multicompartmental
- Ion channels, ligand gates, parameterizable neurotransmitters

### 4. Plasticity & Learning
- Refactor STDP to trait-based system
- Implement triplet STDP, BCM, reward-modulated variants
- Graph-based plasticity for mixed plastic/non-plastic connections
- Neuron model language with trait-based kinetics/plasticity

### 5. Advanced Features
- Astrocytes with tripartite synapses
- Gap junctions and various current models
- Attractor models (ring, grid, Hopfield)
- LSM with reservoirs, readout training

### 6. Neuromodulation & Pathology
- Metabotropic NTs (dopamine, serotonin, etc.)
- Pathology simulations (schizophrenia, hallucinations)
- Virtual medications via receptor modulation
- Cue models, working memory

### 7. Applications
- STDP-based classifiers (MNIST, unsupervised)
- R-STDP classifiers/regressors
- LSM-based cognition (navigation, TSP)
- EEG processing, frequency analysis

### 8. Tools & Interfaces
- Lixirnet EEG tools
- Python bindings for all features
- .nb file parsing for custom models
- Benchmarking suite

## Roadmap Phases

### Phase 1: Foundation (1-2 months) - High Priority
**Goal**: Solidify core infrastructure, basic models, and documentation.

1. **Documentation** (1 week)
   - Add explanations and diagrams to docs/
   - Update README with model breakdowns
   - Effort: Low

2. **Performance Optimizations** (2 weeks)
   - Implement BufWriter capacity changes
   - Optimize lattice dt and allocation
   - Test with Rayon parallelization
   - Effort: Medium

3. **Core Model Completion** (4 weeks)
   - Complete all integrate-and-fire variants with tests
   - Finish HH neurotransmission (AMPA, NMDA, GABA)
   - Implement multicompartmental HH
   - Add .nb parsing for custom models
   - Effort: High

4. **Basic Plasticity Refactor** (2 weeks)
   - Refactor STDP to trait-based system
   - Implement triplet STDP
   - Add plasticity condition checks
   - Effort: Medium

**Milestones**: All backend neuron models tested, basic plasticity working, performance benchmarks.

### Phase 2: Advanced Biology (3-4 months) - Medium Priority
**Goal**: Add biological complexity, attractors, neuromodulation.

1. **Advanced Plasticity** (3 weeks)
   - Implement BCM, reward-modulated plasticity
   - Add R-STDP with dopamine traces
   - Integrate plasticity with graph structures
   - Effort: Medium

2. **Neuromodulation & Pathology** (4 weeks)
   - Add metabotropic NTs (dopamine, serotonin)
   - Implement astrocytes and tripartite synapses
   - Create pathology models (schizophrenia simulation)
   - Virtual medications via receptor modulation
   - Effort: High

3. **Attractors & Memory** (4 weeks)
   - Implement ring attractor for head direction
   - Add Hopfield network with discrete/binary variants
   - Cue models and working memory simulations
   - Fading memory via gap junction decay
   - Effort: High

4. **LSM Basics** (3 weeks)
   - Build LSM reservoir with excitatory/inhibitory balance
   - Implement readout training via R-STDP
   - Stability metrics (eigenvalues, convergence)
   - Effort: Medium

**Milestones**: Full biological neuron models, basic cognition simulations, pathology prototypes.

### Phase 3: Applications & Intelligence (5-6 months) - Low Priority
**Goal**: Develop intelligent systems, classifiers, cognitive models.

1. **Classifiers & Regressors** (6 weeks)
   - STDP-based unsupervised classifier (MNIST)
   - R-STDP classifier/regressor with reward optimization
   - LSM-based classification and regression
   - Effort: High

2. **Cognitive Simulations** (8 weeks)
   - Navigation with head direction/grid attractors
   - Traveling salesman problem solving
   - Hallucination models via noise/modulation
   - Small-world architectures in LSM
   - Effort: High

3. **EEG & Analysis** (4 weeks)
   - Integrate EEG processing in Lixirnet
   - Frequency band analysis (beta/gamma)
   - Fourier transforms, spectral density
   - Effort: Medium

4. **Integration & Polish** (4 weeks)
   - Full Python bindings for all features
   - Benchmarking suite for all models
   - Documentation updates and tutorials
   - Effort: Medium

**Milestones**: End-to-end cognitive tasks, publication-ready models.

### Phase 4: Extensions & Research (6+ months) - Future
**Goal**: Advanced research, new models, community features.

1. **Advanced Research**
   - GPU/accelerated implementations (WGPU, CUDA)
   - Large-scale simulations (multi-region brains)
   - Novel architectures (neuro-astrocytic networks)

2. **Community & Tools**
   - Open-source contributions
   - Educational resources
   - API stability and deprecation management

## Dependencies & Prerequisites
- Rust nightly with edition2024 for compilation
- Fix shnn-micro compilation issues
- Python 3.12+ for bindings (with ABI compatibility)
- Benchmarking tools for performance validation

## Risk Mitigation
- **Technical Risks**: Complex models may require iterative testing; allocate buffer time.
- **Scope Creep**: Stick to phased approach; prioritize based on biological accuracy.
- **Performance**: Monitor and optimize early; use profiling tools.
- **Dependencies**: Ensure nightly Rust stability; have fallback toolchains.

## Success Metrics
- All core neuron models implemented and tested
- Performance benchmarks show efficient simulations
- Prototype cognitive tasks (e.g., navigation) demonstrate intelligence
- Documentation covers all major concepts with examples
- Python interface enables easy experimentation

## Next Steps
1. Update memory bank with this roadmap
2. Assign Phase 1 tasks to immediate TODOs
3. Begin with documentation and performance optimizations