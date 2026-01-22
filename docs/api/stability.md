# API Stability Policy

This document outlines the API stability guarantees and policies for the Spiking Neural Networks (SHNN) library.

## Stability Levels

SHNN APIs are classified into three stability levels:

### Stable ![Stable](https://img.shields.io/badge/stability-stable-green)

Stable APIs are mature, well-tested, and provide strong backward compatibility guarantees. These APIs are recommended for production use.

**Guarantees:**
- No breaking changes in minor or patch releases
- Only additive changes allowed
- Deprecation notices provided for planned removals (12 months minimum deprecation period)
- Full semantic versioning compliance for stable APIs

### Experimental ![Experimental](https://img.shields.io/badge/stability-experimental-yellow)

Experimental APIs are under active development and may change significantly or be removed in future releases. Use with caution.

**Characteristics:**
- May include breaking changes in minor releases
- Subject to redesign based on user feedback
- Documentation may be incomplete
- Not recommended for production use without careful evaluation

### Unstable ![Unstable](https://img.shields.io/badge/stability-unstable-red)

Unstable APIs are in early development and have no stability guarantees. They may be changed or removed at any time without notice.

**Characteristics:**
- Breaking changes possible in any release
- May be removed entirely
- Primarily for internal development or advanced users
- Not suitable for production use

## Classification Criteria

### Stable APIs

- Core neuron models: `IzhikevichNeuron`, `LeakyIntegrateAndFire`, basic integrate-and-fire variants
- Basic network structures: `Lattice`, `LatticeNetwork`
- Fundamental plasticity: `STDP`
- Basic history types: `GridVoltageHistory`, `AverageVoltageHistory`, `SpikeHistory`
- Core traits: `IterateAndSpike`, `LatticeHistory`, `RunLattice`, `RunNetwork`

### Experimental APIs

- Advanced neuron models: `HodgkinHuxleyNeuron`, `MorrisLecarNeuron`, multicompartment neurons
- Advanced plasticity: `BCM`, `RewardModulatedSTDP`, `TripletSTDP`, neuromodulation
- Complex architectures: attractors, liquid state machines, digital twin components
- GPU acceleration: all GPU-related types and functions
- Classifiers and regressors: `STDPClassifier`, `LSMClassifier`, etc.
- Advanced features: EEG processing, fitting algorithms

### Unstable APIs

- Currently none - all public APIs are at least experimental

## Versioning Policy

SHNN follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version increments: Breaking changes to stable APIs
- **MINOR** version increments: New features, potentially breaking experimental APIs
- **PATCH** version increments: Bug fixes, performance improvements, documentation updates

## Deprecation Policy

When a stable API needs to be changed or removed:

1. **Deprecation Notice**: Mark the API as deprecated with clear migration guidance
2. **Grace Period**: Minimum 12 months deprecation period before removal
3. **Communication**: Deprecation notices in release notes, documentation, and compile-time warnings
4. **Migration Support**: Provide migration tools/guides where feasible

## Breaking Changes

Breaking changes to stable APIs require:

1. **Major Version Bump**: Increment major version number
2. **Migration Guide**: Comprehensive guide for upgrading
3. **Advance Notice**: Deprecation in previous minor release where possible
4. **Community Review**: For significant changes, community feedback period

## Experimental API Promotion

Experimental APIs may be promoted to stable status when:

- API design has stabilized based on user feedback
- Comprehensive tests are in place
- Documentation is complete
- No major design issues identified in production use

## Feedback and Contributions

- Report stability issues or API design problems on GitHub
- Participate in API design discussions for experimental features
- Contribute tests and documentation to help stabilize APIs

## Exceptions

- Security fixes may introduce breaking changes without following normal deprecation policies
- Bug fixes that correct unintended behavior may be applied with minimal notice
- Compiler compatibility (e.g., Rust edition updates) may require immediate changes</content>
</xai:function_call">The file docs/api/stability.md was created successfully.