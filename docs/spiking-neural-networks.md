Spiking Neural Networks
Generalized spiking neural network system with various intergrate and fire models as well as Hodgkin Huxley models, EEG processing with fourier transforms, and power spectral density calculations

# Biological Neuron Models Broken Down

This section provides detailed explanations of the neuron models implemented in HSNN, their mathematical formulations, biological underpinnings, and practical applications.

## Leaky Integrate-and-Fire (LIF) Model

The Leaky Integrate-and-Fire model is the simplest spiking neuron model, balancing computational efficiency with basic neural dynamics.

### Mathematical Formulation

The membrane potential \(v\) evolves according to:

\[\tau \frac{dv}{dt} = -(v - v_{\text{rest}}) + R I(t)\]

Where:
- \(\tau\): Membrane time constant
- \(v_{\text{rest}}\): Resting potential
- \(R\): Membrane resistance
- \(I(t)\): Input current

When \(v\) reaches the threshold \(v_{\text{th}}\), the neuron spikes and resets to \(v_{\text{reset}}\).

### Variants in HSNN

- **Basic LIF**: Simple leaky integration without adaptation
- **Adaptive LIF**: Adds spike-frequency adaptation via a recovery variable
- **Adaptive Exponential LIF**: Includes exponential approach to threshold for better spike initiation

### Biological Rationale

LIF models capture the essential spiking behavior observed in many neuron types while being computationally efficient for large-scale simulations.

## Izhikevich Model

The Izhikevich model provides a computationally efficient way to reproduce various spiking patterns observed in cortical neurons.

### Mathematical Formulation

The model uses two variables: membrane potential \(v\) and recovery variable \(u\):

\[\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I\]

\[\frac{du}{dt} = a(bv - u)\]

When \(v \geq 30\) mV, reset:
\[v \leftarrow c, \quad u \leftarrow u + d\]

Parameters \(a, b, c, d\) determine the spiking pattern:
- Regular spiking: \(a=0.02, b=0.2, c=-65, d=8\)
- Intrinsically bursting: \(a=0.02, b=0.2, c=-55, d=4\)
- Chattering: \(a=0.02, b=0.2, c=-50, d=2\)

### Lattice Implementation

In HSNN, Izhikevich neurons form 2D lattices with local connectivity:

```
Neuron Grid (5x5 example):
O---O---O---O---O
|   |   |   |   |
O---O---O---O---O
|   |   |   |   |
O---O---O---O---O
|   |   |   |   |
O---O---O---O---O
|   |   |   |   |
O---O---O---O---O
```

Each neuron connects to neighbors within a specified radius, typically using electrical synapses (gap junctions) by default.

## Hodgkin-Huxley Model

The Hodgkin-Huxley model provides the most biologically detailed representation of neuronal dynamics, modeling ion channel gating and neurotransmission.

### Mathematical Formulation

The membrane potential follows:

\[C_m \frac{dv}{dt} = -\sum I_{\text{ion}} + I_{\text{inj}}\]

Where ion currents include:
- Sodium current: \(I_{\text{Na}} = g_{\text{Na}} m^3 h (v - E_{\text{Na}})\)
- Potassium current: \(I_{\text{K}} = g_{\text{K}} n^4 (v - E_{\text{K}})\)
- Leak current: \(I_{\text{L}} = g_{\text{L}} (v - E_{\text{L}})\)

Gating variables follow:
\[\frac{dm}{dt} = \alpha_m(v)(1-m) - \beta_m(v)m\]

### Ion Channels

HSNN implements voltage-gated channels:
- **Sodium channels**: Fast activation, slow inactivation
- **Potassium channels**: Delayed rectification
- **Calcium channels**: Various types (L, T, N, P/Q)
- **Ligand-gated channels**: Neurotransmitter receptors

### Neurotransmission

Chemical synapses use neurotransmitter kinetics:

\[\frac{d[T]}{dt} = -\frac{[T]}{\tau_T} + \delta(t - t_{\text{spike}}) T_{\text{max}}\]

Receptor binding:
\[\frac{dR}{dt} = \alpha [T](1-R) - \beta R\]

Where \([T]\) is neurotransmitter concentration, \(R\) is receptor occupancy.

## Neurotransmission Adaptations

### Hodgkin-Huxley to Izhikevich Adaptation

The detailed Hodgkin-Huxley neurotransmission system is adapted for Izhikevich models through simplified kinetics:

- **Fixed receptor kinetics**: Parameters \(\alpha, \beta\) are predetermined rather than dynamically adjustable, reducing computational complexity
- **Effective conductance**: Neurotransmitter effects are modeled as modifications to synaptic strength rather than separate gating variables
- **Rationale**: Biological accuracy vs. simulation efficiency tradeoff; fixed kinetics suffice for many cognitive modeling tasks while maintaining 100x+ speedup over full HH models

### Rationale for Kinetics Design

- **Computational efficiency**: Simplified kinetics enable real-time simulations of large networks
- **Biological plausibility**: Core mechanisms (release, diffusion, binding, clearance) are preserved
- **Modularity**: Traits allow custom kinetics implementations
- **Research applications**: Sufficient detail for studying synaptic plasticity, neuromodulation, and network dynamics

## Attractor Networks

Attractor networks provide stable memory states and spatial representations.

### Ring Attractor (Head Direction Cells)

Models head direction in navigation:

```
Ring Network:
  N1 --> N2 --> N3 --> N4
  ^                   |
  |                   v
  N8 <-- N7 <-- N6 <-- N5
```

Weights: \(w_{ij} = A \exp\left(-\frac{(i-j)^2}{2\sigma^2}\right) - k\)

Where \(A\) is excitation strength, \(\sigma\) controls tuning width, \(k\) provides global inhibition.

### Hopfield Network

Associative memory with discrete/binary neurons:

```
Hopfield Energy Landscape:
     _____
    /     \
   /       \
  /         \
 /           \
/_____________\
  Stable States
```

Update rule: \(s_i(t+1) = \sgn\left(\sum_j w_{ij} s_j(t) - \theta_i\right)\)

Learning: \(w_{ij} = \frac{1}{N} \sum_\mu \xi_i^\mu \xi_j^\mu\)

### Applications

- **Spatial navigation**: Grid cells, place cells
- **Working memory**: Persistent activity states
- **Pattern completion**: Noisy input reconstruction

## Diagrams and Visualizations

### Lattice Networks

HSNN uses 2D lattice structures for spatial neuron arrangements:

![Lattice Network](images/lattice.gif)
*Figure 1: Izhikevich neuron lattice showing spiking activity over time. Each dot represents a neuron spike, demonstrating emergent wave patterns.*

Lattice connectivity typically uses local rules:
- **Radius-based**: Connect neurons within Euclidean distance
- **Moore neighborhood**: 8-connected grid
- **Electrical synapses**: Default gap junction coupling
- **Chemical synapses**: Optional neurotransmitter-mediated connections

### Attractor Dynamics

Ring attractor for head direction:

```
Head Direction Ring Attractor:
  0°   45°   90°  135°  180°
   ↑     ↗     →     ↘     ↓
 315° ← 270° ← 225°
   ↖     ↙

Activity bumps propagate smoothly around the ring,
maintaining directional information.
```

### Hopfield Network Energy Landscape

Hopfield networks converge to stored patterns:

```
Energy Landscape:
     _____
    /     \
   /       \
  /         \
 /           \
/_____________\
Stored Pattern 1    Stored Pattern 2

Noisy input → Energy minimization → Stable state
```

### Neurotransmission Pathways

Detailed synaptic transmission:

![Hodgkin-Huxley Neurotransmission](images/hodgkin_huxley_neurotransmission.png)
*Figure 2: Ionotropic and metabotropic neurotransmission pathways in Hodgkin-Huxley neurons, showing receptor kinetics and intracellular signaling.*
## Digital Twin Architecture

HSNN implements a modular digital twin system for simulating brain-like computation at multiple scales. The digital twin framework provides hierarchical organization from individual neurons to interconnected brain regions.

### BrainRegion Trait

The `BrainRegion` trait defines the interface for modular brain components:

```rust
pub trait BrainRegion {
    fn step(&mut self, inputs: &[f32], dt: f32) -> Vec<Spike>;
    fn get_output(&self) -> &[f32];
    fn connect_to(&mut self, other: &mut dyn BrainRegion);
    fn get_region_type(&self) -> RegionType;
}
```

### Modular Brain Regions

- **CorticalModule**: Generic cortical processing with Izhikevich neurons and STDP plasticity
- **HippocampalModule**: Spatial memory with ring attractor for head direction cells
- **LsmModule**: Liquid state machine reservoir for temporal processing and readout
- **ClassifierModule**: STDP-based or R-STDP-based classification/regression

### DigitalTwin Orchestrator

The `DigitalTwin` struct coordinates inter-region communication:

```rust
pub struct DigitalTwin {
    regions: Vec<Box<dyn BrainRegion>>,
    connections: HashMap<(usize, usize), ConnectionType>,
    neuromodulators: NeuromodulatorSystem,
}

impl DigitalTwin {
    pub fn add_region(&mut self, region: Box<dyn BrainRegion>);
    pub fn connect_regions(&mut self, from: usize, to: usize, conn_type: ConnectionType);
    pub fn step(&mut self, dt: f32);
}
```

### Neuromodulation Integration

Digital twin supports dopamine-modulated plasticity for reinforcement learning and neuromodulated learning across regions.

## Classifiers and Regressors

HSNN provides multiple approaches to classification and regression tasks using biologically plausible mechanisms.

### STDP-Based Classifier

Unsupervised classification using spike-timing dependent plasticity:

- **Architecture**: Poisson input neurons → Excitatory lattice → Inhibitory neurons
- **Learning**: STDP on excitatory connections, winner-take-all inhibition
- **Output**: Class assignment based on highest firing rate in readout layer

### R-STDP Classifier/Regressor

Reward-modulated STDP for supervised learning:

- **Reward Function**: User-defined function mapping network state to reward
- **Dopamine Modulation**: R-STDP weights updated based on reward prediction error
- **Applications**: Classification with reward feedback, regression via spike count coding

### LSM-Based Classification

Liquid state machine with trained readout:

- **Reservoir**: Recurrent spiking network with fixed random connections
- **Readout Training**: Linear regression or R-STDP on reservoir outputs
- **Advantages**: Temporal processing, separation of dynamics from learning

## Advanced Plasticity

### BCM Rule

Bienenstock-Cooper-Munro sliding threshold plasticity:

\[\theta_M = \langle y^2 \rangle\]

\[\Delta w = y(x - \theta_M)\]

Where \(\theta_M\) adapts based on postsynaptic activity history.

### Reward-Modulated Plasticity

Dopamine-modulated learning rules:

- **R-STDP**: STDP modulated by dopamine concentration
- **Reward-Modulated BCM**: BCM rule with reward-dependent scaling
- **Trace-Based Learning**: Eligibility traces for delayed reward

### Triplet STDP

Enhanced STDP with temporal dependencies:

- **Triplet Rule**: Considers pre-post-pre and post-pre-post spike triplets
- **Improved Stability**: Better weight convergence and biological plausibility

## Neuromodulation and Pathology

### Metabotropic Neurotransmitters

Slow-acting neuromodulators with intracellular effects:

- **Dopamine**: Reward signaling, modulates plasticity and motivation
- **Serotonin**: Mood regulation, affects excitability and plasticity
- **Acetylcholine**: Attention and learning modulation
- **Glutamate**: Excitotoxicity and synaptic scaling

### Astrocyte Models

Tripartite synapse with glial modulation:

- **Calcium Dynamics**: Astrocyte responds to synaptic glutamate
- **Gliotransmission**: Astrocytes release modulators affecting synaptic strength
- **Neuro-astrocytic Networks**: Coupled neuron-astrocyte systems

### Pathology Simulation

Virtual models of neurological conditions:

- **Schizophrenia**: NMDA/GABA imbalances with hallucination-like misclassifications
- **Alzheimer's**: Amyloid-beta effects on synaptic plasticity
- **Parkinson's**: Dopamine depletion and motor symptoms

### Virtual Medications

Receptor modulation for treatment simulation:

- **Antipsychotics**: NMDA/GABA receptor modulation
- **Antidepressants**: Serotonin reuptake inhibition
- **Stimulants**: Dopamine transporter blockade

## Future Research Directions

- **GPU Acceleration**: Full CUDA/OpenCL support for large-scale simulations
- **Multi-Scale Modeling**: Bridging molecular, cellular, and systems levels
- **Closed-Loop Experiments**: Real-time interaction with physical systems
- **Neuromorphic Hardware**: Deployment on specialized spiking hardware
