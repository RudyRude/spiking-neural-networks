# Contributing to Spiking Neural Networks

Thank you for your interest in contributing to the Spiking Neural Networks project! This document provides guidelines and information for contributors.

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

This pledge is enforced through our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Ways to Contribute

- **Report bugs** by opening issues on GitHub
- **Suggest features** by opening feature requests
- **Improve documentation** by submitting documentation patches
- **Write code** by submitting pull requests
- **Review code** by participating in pull request reviews
- **Test the software** and report issues

### Development Workflow

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `main`
4. **Make your changes** following the coding standards below
5. **Write tests** for your changes
6. **Run the test suite** to ensure everything passes
7. **Commit your changes** with clear, descriptive commit messages
8. **Push your branch** to your fork
9. **Submit a pull request** to the main repository

### Coding Standards

#### Rust Code

- Use the nightly Rust toolchain with unstable edition 2024 (see `backend/rust-toolchain.toml`)
- Follow standard Rust formatting with `rustfmt`
- Use `clippy` for linting
- Implement the `IterateAndSpike` trait with neurotransmitter/receptor kinetics traits for custom neuron models
- Use `raw_create_agent_type_for_lattice!` macro for lattice agent creation
- Lattice structs default to `electrical_synapse=true`, `chemical_synapse=false`
- For GPU lattices, use custom buffer macros: `read_and_set_buffer!`, `write_buffer!`, etc.
- Extensive trait system: implement `NeurotransmitterKinetics`/`ReceptorKinetics` for custom models

#### Python Code

- Python interface requires `maturin develop --release` for development builds (not standard pip)
- Follow PEP 8 style guidelines
- Use type hints where possible
- Write comprehensive docstrings

#### General

- Write clear, self-documenting code
- Add comments for complex algorithms
- Use meaningful variable and function names
- Keep functions small and focused on a single responsibility

### Testing Requirements

- All new code must include unit tests
- Tests must validate neural invariants (e.g., grid_formation_invariant.rs, gpu_accuracy.rs)
- Integration tests for Python CPU interface, CLI tools, end-to-end experiments, and error handling
- GPU accuracy tests for CUDA/OpenCL implementations
- Run full test suite before submitting PR

### Pull Request Process

1. **Ensure your PR description** clearly describes the changes and their purpose
2. **Link related issues** if applicable
3. **Include tests** for new functionality
4. **Update documentation** if necessary
5. **Ensure CI passes** all checks
6. **Request review** from maintainers

### Review Process

- At least one maintainer must approve the PR
- Reviewers may request changes
- Once approved, a maintainer will merge the PR
- Maintainers may squash commits for cleaner history

## Getting Help

- Check existing issues and documentation
- Join our community discussions
- Contact maintainers directly if needed

Thank you for contributing to the Spiking Neural Networks project!