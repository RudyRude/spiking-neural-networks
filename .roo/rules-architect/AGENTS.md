# Project Architecture Rules (Non-Obvious Only)

- Trait-based architecture: IterateAndSpike, NeurotransmitterKinetics, ReceptorKinetics for extensibility
- Lattice networks require explicit ID assignment and connection management
- Electrical synapses enabled by default in lattices (chemical disabled)
- GPU/CPU duality with separate implementations and buffer macros
- Workspace structure excludes embedded and python crates by default