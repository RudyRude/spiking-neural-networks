# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
YYYY-MM-DD HH:MM:SS - Log of updates made will be appended as footnotes to the end of this file.

*

## Project Goal

* Build a comprehensive spiking neural network (SNN) system for simulating brain-like computation, from individual neurons to cognitive tasks, with biological accuracy and applications in AI, neuroscience research, and pathology modeling.

## Key Features

* Biological neuron models: Integrate-and-fire, Izhikevich, Hodgkin-Huxley with neurotransmission
* Plasticity and learning: STDP, R-STDP, neuromodulated learning
* Advanced architectures: Attractors, liquid state machines, Hopfield networks
* Digital twin brain simulation: Modular regions with inter-connectivity
* Applications: Classifiers, cognitive tasks (navigation, memory), pathology modeling
* Tools: EEG processing, Python bindings, benchmarking suite
* Performance: GPU acceleration, parallel processing, optimized memory usage   

## Overall Architecture

* Modular digital twin with BrainRegion traits for brain modules (cortex, hippocampus, LSM)
* Hierarchical SNN system: neurons -> lattices -> networks -> brain regions
* Support for biological models: integrate-and-fire, Izhikevich, Hodgkin-Huxley
* Plasticity and learning: STDP, R-STDP, neuromodulation
* Applications: classifiers, attractors, LSM-based cognition, EEG processing
* Roadmap: See docs/project-roadmap.md for phased development plan