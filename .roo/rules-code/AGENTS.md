# Project Coding Rules (Non-Obvious Only)

- Use nightly Rust toolchain with unstable edition 2024 (backend/rust-toolchain.toml)
- Implement IterateAndSpike trait with neurotransmitter/receptor kinetics traits
- Lattice structs default to electrical_synapse=true, chemical_synapse=false (counterintuitive for neuroscience)
- Use raw_create_agent_type_for_lattice! macro for lattice agent creation
- GPU lattices require custom buffer macros: read_and_set_buffer!, write_buffer!, etc.
- Extensive trait system for neuron models - implement NeurotransmitterKinetics/ReceptorKinetics for custom models