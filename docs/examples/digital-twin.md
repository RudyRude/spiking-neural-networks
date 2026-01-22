# Digital Twin Examples

This tutorial demonstrates how to build and use HSNN's digital twin framework for multi-scale brain simulations.

## Table of Contents

- [Basic Digital Twin Construction](#basic-digital-twin-construction)
- [Multi-Region Brain Model](#multi-region-brain-model)
- [Neuromodulated Cognition](#neuromodulated-cognition)
- [Pathology Simulation](#pathology-simulation)
- [Real-time Interaction](#real-time-interaction)

## Basic Digital Twin Construction

### Simple Two-Region Model

```python
import lixirnet as ln
import numpy as np

# Create cortical region
cortex = ln.CorticalModule(
    name="cortex",
    neurons=1000,
    neuron_type=ln.IzhikevichNeuron,
    plasticity=ln.STDP(learning_rate=0.01),
    connectivity=ln.LocalConnectivity(radius=2)
)

# Create hippocampal region
hippocampus = ln.HippocampalModule(
    name="hippocampus",
    neurons=500,
    attractor_type=ln.RingAttractor(neurons_per_ring=50),
    head_direction_input=True
)

# Create digital twin
brain = ln.DigitalTwin()

# Add regions
brain.add_region(cortex)
brain.add_region(hippocampus)

# Connect regions (cortex -> hippocampus)
brain.connect_regions(
    source="cortex",
    target="hippocampus",
    connection_type=ln.ChemicalSynapse(
        strength=0.5,
        neurotransmitter=ln.Glutamate(),
        delay=0.002  # 2ms delay
    )
)

print("Digital twin created with regions:", brain.list_regions())
```

### Custom Brain Region

```python
class CustomRegion(ln.BrainRegion):
    def __init__(self, name, neurons=100):
        super().__init__(name)
        self.neurons = neurons
        self.voltage = np.zeros(neurons)
        self.spike_history = []

    def step(self, inputs, dt):
        """Custom region dynamics."""
        # Simple leaky integrate-and-fire
        decay = np.exp(-dt / 0.02)  # 20ms time constant
        self.voltage *= decay
        self.voltage += inputs

        # Spike generation
        spikes = self.voltage > 1.0
        self.voltage[spikes] = 0.0  # Reset

        # Record spikes
        spike_times = np.where(spikes)[0]
        for neuron_id in spike_times:
            self.spike_history.append((neuron_id, self.current_time))

        self.current_time += dt
        return spike_times

    def get_output(self):
        """Return current activity level."""
        return np.mean(self.voltage)

    def reset(self):
        """Reset region state."""
        self.voltage = np.zeros(self.neurons)
        self.spike_history = []
        self.current_time = 0.0

# Add custom region
custom = CustomRegion("custom", neurons=200)
brain.add_region(custom)
```

## Multi-Region Brain Model

### Complete Brain Model

```python
def create_brain_model():
    """Create a multi-region brain model."""
    brain = ln.DigitalTwin()

    # Visual cortex
    visual_cortex = ln.CorticalModule(
        name="visual_cortex",
        neurons=2000,
        neuron_type=ln.IzhikevichNeuron,
        plasticity=ln.STDP(),
        input_connectivity=ln.GaussianConnectivity(sigma=1.0)
    )

    # Motor cortex
    motor_cortex = ln.CorticalModule(
        name="motor_cortex",
        neurons=1500,
        neuron_type=ln.IzhikevichNeuron,
        plasticity=ln.STDP(),
        output_connectivity=ln.LocalConnectivity(radius=3)
    )

    # Prefrontal cortex (working memory)
    prefrontal = ln.CorticalModule(
        name="prefrontal",
        neurons=800,
        neuron_type=ln.IzhikevichNeuron,
        plasticity=ln.RewardModulatedSTDP(dopamine_decay=0.9),
        working_memory=True
    )

    # Hippocampus (spatial memory)
    hippocampus = ln.HippocampalModule(
        name="hippocampus",
        neurons=1000,
        place_cells=200,
        head_direction_cells=50,
        grid_cells={"spacing": [40, 40], "orientation": 30}
    )

    # Amygdala (emotional processing)
    amygdala = ln.AmygdalaModule(
        name="amygdala",
        neurons=300,
        fear_conditioning=True,
        reward_sensitivity=0.8
    )

    # Thalamus (sensory relay)
    thalamus = ln.ThalamusModule(
        name="thalamus",
        neurons=600,
        sensory_relay=True,
        sleep_regulation=True
    )

    # Add all regions
    brain.add_region(visual_cortex)
    brain.add_region(motor_cortex)
    brain.add_region(prefrontal)
    brain.add_region(hippocampus)
    brain.add_region(amygdala)
    brain.add_region(thalamus)

    # Define connectivity
    connections = [
        # Sensory flow
        ("thalamus", "visual_cortex", ln.ExcitatorySynapse(0.3)),
        ("thalamus", "amygdala", ln.ExcitatorySynapse(0.2)),

        # Cortical connections
        ("visual_cortex", "prefrontal", ln.ExcitatorySynapse(0.4)),
        ("prefrontal", "motor_cortex", ln.ExcitatorySynapse(0.5)),

        # Memory systems
        ("visual_cortex", "hippocampus", ln.ExcitatorySynapse(0.2)),
        ("hippocampus", "prefrontal", ln.ExcitatorySynapse(0.3)),
        ("prefrontal", "hippocampus", ln.ExcitatorySynapse(0.25)),

        # Emotional modulation
        ("amygdala", "prefrontal", ln.ModulatorySynapse(0.1, ln.Dopamine())),
        ("amygdala", "hippocampus", ln.ModulatorySynapse(0.15, ln.Norepinephrine())),

        # Motor output
        ("motor_cortex", "thalamus", ln.ExcitatorySynapse(0.2)),
    ]

    for source, target, conn_type in connections:
        brain.connect_regions(source, target, conn_type)

    return brain

# Create and initialize brain model
brain = create_brain_model()
brain.initialize()  # Set up internal state

print("Brain model created with regions:")
for region_name in brain.list_regions():
    region = brain.get_region(region_name)
    print(f"  {region_name}: {region.neuron_count} neurons")
```

### Hierarchical Connectivity

```python
# Define brain areas hierarchically
brain_areas = {
    'sensory': ['visual_cortex', 'auditory_cortex', 'somatosensory_cortex'],
    'association': ['prefrontal_cortex', 'parietal_cortex', 'temporal_cortex'],
    'motor': ['primary_motor_cortex', 'premotor_cortex', 'supplementary_motor_cortex'],
    'limbic': ['hippocampus', 'amygdala', 'hypothalamus'],
    'subcortical': ['thalamus', 'basal_ganglia', 'cerebellum']
}

# Create regions for each area
regions = {}
for area, subregions in brain_areas.items():
    for subregion in subregions:
        # Create appropriate module type
        if 'cortex' in subregion:
            module = ln.CorticalModule(
                name=subregion,
                neurons=np.random.randint(500, 2000),
                hierarchical_level=area
            )
        elif 'hippocampus' in subregion:
            module = ln.HippocampalModule(
                name=subregion,
                neurons=800,
                place_cells=150
            )
        # ... other region types

        regions[subregion] = module

# Add hierarchical connections
for area, subregions in brain_areas.items():
    # Connect within area (dense)
    for i, region1 in enumerate(subregions):
        for region2 in subregions[i+1:]:
            brain.connect_regions(
                region1, region2,
                ln.HierarchicalSynapse(strength=0.3, level=area)
            )

    # Connect between areas (sparse, specific pathways)
    if area == 'sensory':
        for sensory_region in subregions:
            for assoc_region in brain_areas['association']:
                brain.connect_regions(
                    sensory_region, assoc_region,
                    ln.FeedforwardSynapse(strength=0.2)
                )
```

## Neuromodulated Cognition

### Working Memory Task

```python
# Create prefrontal cortex with working memory
prefrontal = ln.PrefrontalModule(
    name="prefrontal_cortex",
    neurons=1000,
    working_memory_capacity=5,  # Remember 5 items
    dopamine_modulation=True,
    noradrenergic_attention=True
)

# Set up working memory task
task = ln.WorkingMemoryTask(
    sequence_length=4,
    delay_period=2.0,  # 2 second delay
    distractors=True,  # Add distracting inputs during delay
    reward_schedule=ln.ProgressiveRatio(reward_increase=1.5)
)

# Run simulation
results = brain.simulate_task(
    task=task,
    duration=10.0,
    record_regions=['prefrontal_cortex']
)

# Analyze performance
memory_accuracy = results.calculate_memory_accuracy()
distraction_resistance = results.measure_distraction_resistance()

print(f"Memory accuracy: {memory_accuracy:.3f}")
print(f"Distraction resistance: {distraction_resistance:.3f}")
```

### Decision Making with Dopamine

```python
# Create decision network
decision_network = ln.DecisionModule(
    name="decision_cortex",
    neurons=800,
    options=3,  # 3 choice options
    reward_sensitivity=0.7,
    exploration_rate=0.1
)

# Add neuromodulators
dopamine = ln.DopamineSystem(
    baseline=0.2,
    phasic_response=True,
    tonic_control=True
)

brain.add_neuromodulator(dopamine)

# Two-armed bandit task
bandit = ln.MultiArmedBandit(
    arms=3,
    reward_probabilities=[0.8, 0.5, 0.2],  # Different reward rates
    reward_magnitude=1.0,
    trials=100
)

# Run decision making simulation
decisions = []
rewards = []
dopamine_levels = []

for trial in range(bandit.trials):
    # Make decision
    choice = brain.decide(bandit.get_state())
    decisions.append(choice)

    # Get reward
    reward = bandit.pull_arm(choice)
    rewards.append(reward)

    # Update dopamine
    brain.deliver_reward(reward, choice)

    # Record dopamine level
    dopamine_levels.append(brain.get_neuromodulator_level('dopamine'))

# Analyze learning
optimal_choice_rate = np.mean([
    1 if choice == np.argmax(bandit.reward_probabilities) else 0
    for choice in decisions[-50:]  # Last 50 trials
])

print(f"Optimal choice rate: {optimal_choice_rate:.3f}")
```

## Pathology Simulation

### Schizophrenia Model

```python
# Create schizophrenia pathology model
schizophrenia = ln.PathologyModel.schizophrenia(
    nmda_hypofunction=0.4,      # 40% NMDA reduction
    gaba_imbalance=0.3,         # GABA interneuron dysfunction
    dopamine_hyperactivity=0.5, # Mesolimbic dopamine excess
    glutamate_cortical_deficit=0.25  # Cortical glutamate deficit
)

# Apply to brain model
brain.apply_pathology(schizophrenia, regions=['prefrontal_cortex', 'hippocampus'])

# Simulate positive symptoms (hallucinations)
hallucination_task = ln.HallucinationInductionTask(
    sensory_input_pattern='visual_grid',
    reduced_inhibition=True,
    aberrant_salience=True
)

hallucination_results = brain.simulate_task(hallucination_task, duration=5.0)

# Analyze hallucination patterns
false_positives = hallucination_results.detect_false_perceptions()
aberrant_synchrony = hallucination_results.measure_abnormal_synchrony()

print(f"False positive rate: {false_positives:.3f}")
print(f"Aberrant synchrony index: {aberrant_synchrony:.3f}")
```

### Alzheimer's Disease Simulation

```python
# Alzheimer's pathology
alzheimers = ln.PathologyModel.alzheimers(
    amyloid_beta_load=0.7,      # Amyloid plaque burden
    tau_hyperphosphorylation=0.6, # Tau tangle formation
    cholinergic_deficit=0.5,    # Acetylcholine reduction
    synaptic_loss=0.4          # Synaptic density reduction
)

# Apply pathology
brain.apply_pathology(alzheimers, regions=['hippocampus', 'entorhinal_cortex'])

# Memory task with pathology
memory_task = ln.EpisodicMemoryTask(
    encoding_phase=2.0,    # 2s encoding
    delay_phase=10.0,      # 10s delay
    retrieval_phase=2.0,   # 2s retrieval
    interference=True      # Proactive interference
)

memory_results = brain.simulate_task(memory_task)

# Analyze memory deficits
recall_accuracy = memory_results.calculate_recall_accuracy()
false_memories = memory_results.count_false_memories()
consolidation_deficit = memory_results.measure_consolidation_deficit()

print(f"Recall accuracy: {recall_accuracy:.3f}")
print(f"False memories: {false_memories}")
print(f"Consolidation deficit: {consolidation_deficit:.3f}")
```

### Virtual Medication

```python
# Simulate antipsychotic treatment
antipsychotic = ln.VirtualMedication.antipsychotic(
    target='dopamine_d2',       # D2 receptor antagonism
    affinity=0.8,               # High affinity
    brain_penetration=0.6,      # Good BBB penetration
    side_effect_profile={
        'extrapyramidal': 0.3,  # EPS risk
        'sedation': 0.4,        # Sedation
        'weight_gain': 0.5      # Metabolic effects
    }
)

# Administer medication
brain.administer_medication(antipsychotic, dose=1.0)

# Simulate treatment response
treatment_results = brain.simulate_treatment_response(
    duration=30.0,  # 30 days equivalent
    symptom_assessment='positive_negative_symptoms'
)

# Analyze efficacy and side effects
symptom_reduction = treatment_results.calculate_symptom_reduction()
side_effects = treatment_results.assess_side_effects()

print(f"Positive symptom reduction: {symptom_reduction['positive']:.1%}")
print(f"Negative symptom reduction: {symptom_reduction['negative']:.1%}")
print(f"EPS incidence: {side_effects['extrapyramidal']:.1%}")
```

## Real-time Interaction

### Closed-Loop Experiment

```python
import time

# Create real-time brain model
brain = ln.RealtimeDigitalTwin()
brain.add_region(ln.CorticalModule("motor_cortex", neurons=500))
brain.add_region(ln.CorticalModule("sensory_cortex", neurons=500))

# Connect with feedback
brain.connect_regions(
    "sensory_cortex", "motor_cortex",
    ln.BidirectionalSynapse(strength=0.3, delay=0.01)
)

# Real-time simulation loop
brain.start_realtime_simulation()

try:
    for step in range(1000):  # 10 seconds at 100 Hz
        # Get external input (e.g., from sensors)
        sensory_input = get_sensor_data()

        # Step brain model
        motor_output = brain.step_realtime(sensory_input)

        # Apply motor output to actuators
        apply_motor_commands(motor_output)

        # Small delay to maintain real-time
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Real-time simulation stopped")

finally:
    brain.stop_realtime_simulation()
```

### Adaptive Learning

```python
# Create adaptive brain model
adaptive_brain = ln.AdaptiveDigitalTwin()

# Add plasticity and neuromodulation
adaptive_brain.add_plasticity(ln.MetaPlasticity())  # Plasticity of plasticity
adaptive_brain.add_neuromodulator(ln.NorepinephrineSystem())  # Attention modulation

# Learning task
learning_task = ln.ContinuousLearningTask(
    task_complexity='increasing',
    feedback_type='reinforcement',
    adaptation_rate=0.1
)

# Adaptive simulation
performance_history = []

for episode in range(100):
    # Run episode
    result = adaptive_brain.run_episode(learning_task)

    # Record performance
    performance_history.append(result.performance)

    # Adapt brain parameters based on performance
    if result.performance < 0.5:
        # Increase learning rate
        adaptive_brain.adjust_learning_rate(1.1)
    elif result.performance > 0.8:
        # Fine-tune plasticity
        adaptive_brain.optimize_plasticity()

    # Update neuromodulators
    adaptive_brain.update_neuromodulators(result.emotional_state)

print(f"Final performance: {performance_history[-1]:.3f}")
print(f"Learning improvement: {performance_history[-1] - performance_history[0]:.3f}")