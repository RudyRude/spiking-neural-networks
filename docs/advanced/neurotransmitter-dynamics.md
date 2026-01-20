# Advanced Tutorial: Neurotransmitter Dynamics

This tutorial explores the biochemical signaling mechanisms in spiking neural networks, focusing on neurotransmitter release, diffusion, and receptor dynamics.

## Introduction to Neurotransmitter Systems

Neurotransmitters are chemical messengers that mediate synaptic transmission. The framework implements detailed models of neurotransmitter dynamics including release, diffusion, reuptake, and receptor binding.

## Basic Neurotransmitter Types

The framework supports major neurotransmitter systems found in biological neural networks.

### Glutamate (Excitatory)

Glutamate is the primary excitatory neurotransmitter in the brain.

```python
import lixirnet as ln
import numpy as np

# Create lattice with glutamate synapses
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 10, 10)

# Define glutamate neurotransmitter
glutamate = ln.GlutamateNeurotransmitter(
    release_probability=0.5,     # Release probability on spike
    quantum_size=5000,           # Vesicles per release
    diffusion_coefficient=0.1,   # Diffusion rate
    reuptake_rate=0.01,          # Reuptake time constant
    initial_concentration=0.0    # Initial extracellular concentration
)

# AMPA receptor (fast excitatory)
ampa_receptor = ln.AMPAReceptor(
    binding_rate=0.1,      # Binding rate constant
    unbinding_rate=0.05,   # Unbinding rate constant
    conductance=10.0,      # Peak conductance (nS)
    reversal_potential=0.0  # Reversal potential (mV)
)

# Connect with neurotransmitter dynamics
lattice.connect(
    lambda x, y: True,  # All-to-all connectivity
    neurotransmitter=glutamate,
    receptor=ampa_receptor
)
```

### GABA (Inhibitory)

GABA is the primary inhibitory neurotransmitter.

```python
# GABA neurotransmitter
gaba = ln.GABANeurotransmitter(
    release_probability=0.3,
    quantum_size=3000,
    diffusion_coefficient=0.08,
    reuptake_rate=0.02,
    initial_concentration=0.0
)

# GABA_A receptor (fast inhibitory)
gaba_a = ln.GABA_A_Receptor(
    binding_rate=0.2,
    unbinding_rate=0.1,
    conductance=-5.0,           # Negative conductance (inhibitory)
    reversal_potential=-70.0    # Hyperpolarizing reversal potential
)
```

### Dopamine (Modulatory)

Dopamine plays key roles in reward, motivation, and motor control.

```python
# Dopamine neurotransmitter
dopamine = ln.DopamineNeurotransmitter(
    release_probability=0.1,      # Lower release probability
    quantum_size=1000,
    diffusion_coefficient=0.05,   # Slower diffusion
    reuptake_rate=0.005,          # Slower reuptake (volume transmission)
    initial_concentration=0.0
)

# D1-like receptor (excitatory modulation)
d1_receptor = ln.D1_Receptor(
    binding_rate=0.01,      # Slower binding
    unbinding_rate=0.001,   # Very slow unbinding
    modulation_factor=1.5,  # Increases excitability
    target='ampa_conductance'  # Modulates AMPA receptors
)

# D2-like receptor (inhibitory modulation)
d2_receptor = ln.D2_Receptor(
    binding_rate=0.01,
    unbinding_rate=0.001,
    modulation_factor=0.7,   # Decreases excitability
    target='ampa_conductance'
)
```

## Volume Transmission

Unlike traditional synaptic transmission, some neurotransmitters (like dopamine) use volume transmission - diffusing through extracellular space to affect distant neurons.

```python
# Setup volume transmission
volume_transmitter = ln.VolumeTransmitter(
    neurotransmitter=dopamine,
    diffusion_radius=5,          # Diffusion range (grid units)
    decay_rate=0.01,            # Exponential decay
    boundary_conditions='absorbing'  # Concentration at boundaries
)

# Apply to lattice
lattice.set_volume_transmitter(volume_transmitter)
```

## Multi-Neurotransmitter Synapses

Biological synapses often release multiple neurotransmitters.

```python
# Co-release of glutamate and neuropeptides
co_release_synapse = ln.MultiNeurotransmitterSynapse([
    (glutamate, ampa_receptor, 0.8),    # 80% glutamate
    (ln.SubstanceP(), ln.NK1_Receptor(), 0.2)  # 20% substance P
])

lattice.connect(
    distance_connection,
    synapse_model=co_release_synapse
)
```

## Receptor Dynamics

### Ionotropic Receptors

Fast, ligand-gated ion channels.

```python
# NMDA receptor (voltage-dependent)
nmda = ln.NMDA_Receptor(
    binding_rate=0.05,
    unbinding_rate=0.01,
    conductance=5.0,
    reversal_potential=0.0,
    mg_block_voltage=-30.0,    # Mg²⁺ block voltage
    mg_concentration=1.0       # Mg²⁺ concentration (mM)
)

# Glycine receptor (co-agonist for NMDA)
glycine = ln.GlycineReceptor(
    binding_rate=0.1,
    unbinding_rate=0.02
)
```

### Metabotropic Receptors

Slower, G-protein coupled receptors that trigger intracellular cascades.

```python
# mGluR (metabotropic glutamate receptor)
mglur = ln.mGluR_Receptor(
    binding_rate=0.001,        # Slow binding
    unbinding_rate=0.0001,     # Very slow unbinding
    g_protein_activation=0.1,  # G-protein activation rate
    second_messenger='ip3',    # IP3 signaling pathway
    modulation_target='ampa_conductance',
    modulation_factor=0.8      # Decreases AMPA conductance
)
```

## Plasticity and Neuromodulation

Neurotransmitters can modulate synaptic plasticity.

```python
# Dopamine-modulated STDP
dopamine_stdp = ln.DopamineModulatedSTDP(
    base_stdp=ln.STDPRule(),
    dopamine_sensitivity=2.0,    # How much dopamine affects plasticity
    modulation_window=50.0       # Time window for modulation (ms)
)

# Reward timing affects plasticity
reward_signal = ln.RewardSignal(
    reward_value=1.0,
    timing_precision=10.0,      # Precision of reward timing
    eligibility_trace_tau=100.0 # Eligibility trace time constant
)

lattice.set_reward_system(reward_signal)
```

## Simulating Neurotransmitter Dynamics

### Basic Simulation

```python
# Setup lattice with glutamate synapses
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 5, 5)

glutamate_system = ln.NeurotransmitterSystem(
    neurotransmitter=glutamate,
    receptor=ampa_receptor,
    synapse_model='classic'
)

lattice.connect(distance_connection, neurotransmitter_system=glutamate_system)

# Enable concentration tracking
lattice.track_concentrations = True
lattice.track_currents = True

# Run simulation
lattice.reset_timing()
lattice.reset_history()

lattice.run_lattice(2000)

# Analyze results
concentrations = lattice.get_concentration_history()
currents = lattice.get_current_history()
```

### Visualizing Dynamics

```python
import matplotlib.pyplot as plt

def plot_neurotransmitter_dynamics(lattice):
    """Plot concentration and current over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot concentration
    concentrations = np.array(lattice.get_concentration_history())
    ax1.plot(concentrations.mean(axis=1), label='Mean extracellular [Glu]')
    ax1.set_ylabel('Glutamate Concentration (μM)')
    ax1.set_title('Neurotransmitter Concentration Dynamics')
    ax1.legend()

    # Plot synaptic currents
    currents = np.array(lattice.get_current_history())
    ax2.plot(currents.mean(axis=1), label='Mean synaptic current')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (nA)')
    ax2.set_title('Synaptic Current Dynamics')
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_neurotransmitter_dynamics(lattice)
```

### Receptor Kinetics

```python
def plot_receptor_states(lattice):
    """Plot receptor binding states"""
    receptor_states = lattice.get_receptor_states()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Bound receptors
    axes[0,0].plot(receptor_states['bound_fraction'])
    axes[0,0].set_title('Fraction of Bound Receptors')
    axes[0,0].set_ylabel('Bound Fraction')

    # Open channels
    axes[0,1].plot(receptor_states['open_fraction'])
    axes[0,1].set_title('Fraction of Open Channels')

    # Current through channels
    axes[1,0].plot(receptor_states['current'])
    axes[1,0].set_title('Receptor-Mediated Current')
    axes[1,0].set_ylabel('Current (nA)')
    axes[1,0].set_xlabel('Time (ms)')

    # Desensitization
    axes[1,1].plot(receptor_states['desensitized_fraction'])
    axes[1,1].set_title('Desensitized Receptors')
    axes[1,1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()

plot_receptor_states(lattice)
```

## Advanced Features

### Autoreceptors

Autoreceptors provide feedback regulation of neurotransmitter release.

```python
# GABA_B autoreceptor
gaba_b_autoreceptor = ln.GABA_B_Autoreceptor(
    binding_rate=0.01,
    unbinding_rate=0.001,
    inhibition_factor=0.7,  # Reduces release probability
    presynaptic_location=True
)

# Add to neurotransmitter system
gaba_system = ln.NeurotransmitterSystem(
    neurotransmitter=gaba,
    receptor=gaba_a,
    autoreceptor=gaba_b_autoreceptor
)
```

### Transporter Dynamics

Active reuptake mechanisms.

```python
# Glutamate transporter
glutamate_transporter = ln.GlutamateTransporter(
    v_max=1.0,          # Maximum transport rate
    k_m=10.0,           # Michaelis constant
    energy_cost=1.0,    # ATP consumption
    localization='perisynaptic'  # Location relative to synapse
)

lattice.add_transporter(glutamate_transporter)
```

### Second Messenger Systems

Intracellular signaling cascades.

```python
# Calcium second messenger
calcium_system = ln.SecondMessengerSystem(
    calcium=ln.CalciumDynamics(
        resting_concentration=0.1,    # Resting [Ca²⁺] (μM)
        influx_rate=0.5,              # Ca²⁺ entry rate
        extrusion_rate=0.1,           # Ca²⁺ removal rate
        buffer_capacity=0.1           # Ca²⁺ buffering
    ),
    ip3=ln.IP3Dynamics(),              # IP3 signaling
    camp=ln.cAMPDynamics()             # cAMP signaling
)

lattice.set_second_messenger_system(calcium_system)
```

## Applications

### Neurological Disorders

Model neurotransmitter imbalances:

```python
# Parkinson's disease (dopamine deficiency)
parkinsons_model = ln.NeurologicalDisorderModel(
    neurotransmitter_deficit={
        'dopamine': 0.3,  # 70% dopamine loss
    },
    compensatory_mechanisms=['receptor_upregulation']
)

lattice.apply_disorder_model(parkinsons_model)
```

### Pharmacological Interventions

```python
# SSRI antidepressant (serotonin reuptake inhibitor)
ssri = ln.PharmacologicalAgent(
    target='serotonin_transporter',
    mechanism='inhibition',
    potency=0.8,
    time_course='chronic'
)

lattice.apply_pharmacology(ssri)
```

## Performance Considerations

1. **Computational Cost**: Detailed kinetics increase simulation time
2. **Memory Usage**: Tracking concentrations and states requires memory
3. **Numerical Stability**: Stiff differential equations may need small time steps
4. **Parallelization**: Volume transmission benefits from spatial decomposition

## Troubleshooting

### Common Issues

- **No Response**: Check release probabilities and receptor parameters
- **Oscillations**: Reduce diffusion rates or increase reuptake
- **Memory Issues**: Disable concentration tracking for large lattices
- **Numerical Instability**: Use implicit integration methods

### Validation

```python
# Validate neurotransmitter concentrations
concentrations = lattice.get_concentrations()
assert np.all(concentrations >= 0), "Negative concentrations detected"
assert np.all(concentrations < 1000), "Unrealistic concentration values"

# Check receptor conservation
bound = receptor_states['bound']
unbound = receptor_states['unbound']
total = bound + unbound
assert np.allclose(total, total[0]), "Receptor conservation violated"
```

## Next Steps

- Learn about [GPU Acceleration](gpu-usage.md) for fast neurotransmitter simulations
- Explore [Custom Neuron Models](custom-neuron-models.md)
- See neurotransmitter examples in the [Examples Gallery](../examples/gallery.md)