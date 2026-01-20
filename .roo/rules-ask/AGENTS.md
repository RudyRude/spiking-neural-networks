# Project Documentation Rules (Non-Obvious Only)

- Primary documentation in backend/README.md with usage examples
- Python interface examples in interface/examples/ and experiments/
- Architecture overview: Rust backend with PyO3 bindings via maturin
- Neuron models require trait implementations for custom dynamics (not obvious from file structure)
- GPU support via CUDA/OpenCL, see gpu_impl/ and interface_gpu/