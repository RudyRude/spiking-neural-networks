# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Use nightly Rust toolchain with unstable edition 2024 (backend/rust-toolchain.toml)
- Python interface requires `maturin develop --release` for development builds (not standard pip)
- Lattice structs default to electrical_synapse=true, chemical_synapse=false
- Implement IterateAndSpike trait with neurotransmitter/receptor kinetics traits
- GPU support uses CUDA/OpenCL with custom buffer macros (read_and_set_buffer!, write_buffer!)
- Exported macros: raw_create_agent_type_for_lattice! and raw_create_agent_type_for_network!
- Tests validate neural invariants (grid_formation_invariant.rs, gpu_accuracy.rs)