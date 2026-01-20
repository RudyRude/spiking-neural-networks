# Project Debug Rules (Non-Obvious Only)

- Run specific invariant tests: cargo test gpu_accuracy, cargo test grid_formation_invariant
- GPU lattice debugging requires CUDA/OpenCL setup and may need buffer inspection macros
- Lattice network connections can fail silently if IDs mismatch or synapse types misconfigured
- Neurotransmitter dynamics debug: check concentrations in iterate_and_spike implementations