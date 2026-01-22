# Plasticity Examples

This tutorial demonstrates HSNN's synaptic plasticity mechanisms, including STDP, BCM, reward-modulated learning, and advanced plasticity rules.

## Table of Contents

- [Basic STDP Learning](#basic-stdp-learning)
- [Advanced Plasticity Rules](#advanced-plasticity-rules)
- [Reward-Modulated Plasticity](#reward-modulated-plasticity)
- [Homeostatic Plasticity](#homeostatic-plasticity)
- [Meta-Plasticity](#meta-plasticity)
- [Structural Plasticity](#structural-plasticity)

## Basic STDP Learning

### Simple STDP Training

```python
import lixirnet as ln
import numpy as np
import matplotlib.pyplot as plt

# Create simple network with STDP
network = ln.Network(
    neurons=100,
    connectivity=0.3,
    dt=0.001
)

# Add STDP plasticity
stdp_rule = ln.STDP(
    a_plus=0.01,      # LTP amplitude
    a_minus=0.01,     # LTD amplitude
    tau_plus=20e-3,   # LTP time constant
    tau_minus=20e-3,  # LTD time constant
    w_max=1.0,        # Maximum weight
    w_min=0.0         # Minimum weight
)

network.set_plasticity(stdp_rule)

# Training patterns (pre-post spike pairs)
def generate_stdp_patterns():
    patterns = []

    # LTP: pre before post
    for i in range(10):
        pre_time = 0.01 * i
        post_time = pre_time + 0.005  # 5ms delay
        patterns.append((pre_time, post_time))

    # LTD: post before pre
    for i in range(10):
        post_time = 0.01 * i
        pre_time = post_time + 0.005  # 5ms delay
        patterns.append((pre_time, post_time))

    return patterns

# Training loop
weights_history = []

for epoch in range(50):
    patterns = generate_stdp_patterns()

    for pre_time, post_time in patterns:
        # Create input spikes
        pre_spike = ln.Spike(neuron_id=0, time=pre_time)
        post_spike = ln.Spike(neuron_id=1, time=post_time)

        # Simulate single pattern
        network.reset()
        network.process_spike(pre_spike)
        network.process_spike(post_spike)

        # Run brief simulation for STDP update
        network.step(dt=0.01)

    # Record weight evolution
    weights = network.get_weights()
    weights_history.append(weights[0, 1])  # Weight between neurons 0 and 1

# Plot STDP learning curve
plt.figure(figsize=(10, 6))
plt.plot(weights_history)
plt.axhline(y=0.5, color='r', linestyle='--', label='Initial weight')
plt.xlabel('Epoch')
plt.ylabel('Synaptic Weight')
plt.title('STDP Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final weight: {weights_history[-1]:.3f}")
```

### STDP Window Visualization

```python
# Visualize STDP learning window
def plot_stdp_window(stdp_rule, time_range=(-50e-3, 50e-3), dt=1e-3):
    """Plot the STDP learning window."""
    times = np.arange(time_range[0], time_range[1], dt)
    weight_changes = []

    for delta_t in times:
        if delta_t > 0:
            # Pre before post (LTP)
            change = stdp_rule.calculate_weight_change(delta_t, rule_type='ltp')
        else:
            # Post before pre (LTD)
            change = stdp_rule.calculate_weight_change(-delta_t, rule_type='ltd')

        weight_changes.append(change)

    plt.figure(figsize=(10, 6))
    plt.plot(times * 1000, weight_changes)  # Convert to ms
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Δt (ms)')
    plt.ylabel('Weight Change')
    plt.title('STDP Learning Window')
    plt.grid(True)
    plt.show()

# Plot learning window
stdp = ln.STDP(a_plus=0.01, a_minus=0.01, tau_plus=20e-3, tau_minus=20e-3)
plot_stdp_window(stdp)
```

## Advanced Plasticity Rules

### BCM Rule

```python
# BCM sliding threshold plasticity
bcm_rule = ln.BCMRule(
    learning_rate=0.001,
    theta_tau=1000,     # Threshold time constant
    theta_0=0.5         # Initial threshold
)

network = ln.Network(neurons=200, connectivity=0.1)
network.set_plasticity(bcm_rule)

# BCM requires postsynaptic activity history
def train_bcm(network, input_patterns, epochs=100):
    """Train network with BCM rule."""
    theta_history = []

    for epoch in range(epochs):
        epoch_activity = []

        for pattern in input_patterns:
            # Present pattern
            network.set_inputs(pattern)
            network.step(dt=0.01)

            # Record postsynaptic activity
            activity = network.get_neuron_activities()
            epoch_activity.append(activity)

        # Update BCM threshold based on activity history
        mean_activity = np.mean(epoch_activity, axis=0)
        network.update_bcm_threshold(mean_activity)

        theta_history.append(np.mean(network.get_bcm_thresholds()))

    return theta_history

# Generate input patterns
patterns = [np.random.randn(200) > 0 for _ in range(20)]  # Binary patterns

# Train
theta_history = train_bcm(network, patterns)

# Plot threshold evolution
plt.figure(figsize=(8, 5))
plt.plot(theta_history)
plt.xlabel('Epoch')
plt.ylabel('Mean BCM Threshold')
plt.title('BCM Threshold Evolution')
plt.grid(True)
plt.show()
```

### Triplet STDP

```python
# Triplet STDP with temporal dependencies
triplet_stdp = ln.TripletSTDP(
    a_plus=0.001,      # LTP for pre-post pairs
    a_minus=0.001,     # LTD for post-pre pairs
    a_plus_triplet=0.0005,   # Enhancement for pre-post-pre triplets
    a_minus_triplet=0.0005,  # Enhancement for post-pre-post triplets
    tau_plus=20e-3,
    tau_minus=20e-3,
    tau_x=200e-3,      # Triplet time constant for pre
    tau_y=200e-3       # Triplet time constant for post
)

network = ln.Network(neurons=100, connectivity=0.2)
network.set_plasticity(triplet_stdp)

# Generate triplet spike patterns
def generate_triplet_patterns():
    patterns = []

    # Pre-post-pre triplet (facilitates LTP)
    patterns.append([
        ln.Spike(0, 0.010),   # Pre
        ln.Spike(1, 0.015),   # Post
        ln.Spike(0, 0.020),   # Pre again
    ])

    # Post-pre-post triplet (facilitates LTD)
    patterns.append([
        ln.Spike(1, 0.010),   # Post
        ln.Spike(0, 0.015),   # Pre
        ln.Spike(1, 0.020),   # Post again
    ])

    return patterns

# Training with triplets
weights_over_time = []

for epoch in range(30):
    patterns = generate_triplet_patterns()

    for pattern in patterns:
        network.reset()

        for spike in pattern:
            network.process_spike(spike)
            network.step(dt=0.005)  # Small steps for precise timing

    weights = network.get_weights()
    weights_over_time.append(np.mean(weights))

plt.figure(figsize=(8, 5))
plt.plot(weights_over_time)
plt.xlabel('Epoch')
plt.ylabel('Mean Synaptic Weight')
plt.title('Triplet STDP Learning')
plt.grid(True)
plt.show()
```

## Reward-Modulated Plasticity

### R-STDP Implementation

```python
# Reward-modulated STDP
rstdp = ln.RewardModulatedSTDP(
    base_stdp=ln.STDP(a_plus=0.01, a_minus=0.01),
    dopamine_decay=0.9,        # Dopamine time constant
    reward_window=0.05,        # 50ms reward integration
    eligibility_trace_tau=1.0  # Eligibility trace decay
)

network = ln.Network(neurons=300, connectivity=0.15)
network.set_plasticity(rstdp)

# Add dopamine system
dopamine_system = ln.DopamineSystem(
    baseline=0.1,
    phasic_response=True
)
network.add_neuromodulator(dopamine_system)

# Reinforcement learning task
class SimpleRLTask:
    def __init__(self):
        self.state = 0
        self.target_state = 5

    def get_state(self):
        """Convert state to neural input."""
        input_pattern = np.zeros(300)
        input_pattern[self.state * 50 : (self.state + 1) * 50] = 1.0
        return input_pattern

    def step(self, action):
        """Take action and return reward."""
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Move right
            self.state = min(9, self.state + 1)

        # Reward for reaching target
        reward = 1.0 if self.state == self.target_state else 0.0
        done = self.state == self.target_state

        return reward, done

# Training loop
task = SimpleRLTask()
rewards_history = []
success_rate = []

for episode in range(200):
    episode_reward = 0
    done = False
    step_count = 0

    while not done and step_count < 50:
        # Get current state input
        state_input = task.get_state()
        network.set_inputs(state_input)

        # Run network
        network.step(dt=0.01)

        # Get action from network activity
        activities = network.get_neuron_activities()
        action = np.argmax(activities[200:250])  # Action neurons

        # Take action
        reward, done = task.step(action)
        episode_reward += reward

        # Deliver reward to dopamine system
        network.deliver_reward(reward)

        step_count += 1

    rewards_history.append(episode_reward)
    success_rate.append(1.0 if done else 0.0)

    # Reset task
    task.state = 0

# Plot learning curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.convolve(success_rate, np.ones(10)/10, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Success Rate (smoothed)')
plt.title('Success Rate')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Actor-Critic Architecture

```python
# Actor-critic network with separate value and policy learning
actor_critic = ln.ActorCriticNetwork(
    state_size=10,
    action_size=3,
    critic_learning_rate=0.01,
    actor_learning_rate=0.005,
    discount_factor=0.95
)

# Training loop
for episode in range(500):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Get action from actor
        action_probs = actor_critic.get_action_probabilities(state)
        action = np.random.choice(len(action_probs), p=action_probs)

        # Take action
        next_state, reward, done = env.step(action)

        # Update critic (value function)
        td_error = actor_critic.update_critic(state, reward, next_state, done)

        # Update actor (policy)
        actor_critic.update_actor(state, action, td_error)

        state = next_state
        episode_reward += reward

    print(f"Episode {episode}: Reward = {episode_reward}")
```

## Homeostatic Plasticity

### Synaptic Scaling

```python
# Homeostatic synaptic scaling
homeostatic = ln.HomeostaticPlasticity(
    target_activity=0.1,      # Target firing rate
    scaling_rate=0.001,       # Scaling speed
    activity_window=1000      # Integration window (steps)
)

network = ln.Network(neurons=500, connectivity=0.2)
network.set_plasticity(homeostatic)

# Training with homeostatic regulation
activity_history = []

for epoch in range(100):
    # Run network with random inputs
    network.reset()
    epoch_activity = []

    for step in range(1000):
        random_input = np.random.randn(500) * 0.1
        network.set_inputs(random_input)
        network.step(dt=0.001)

        # Record activity
        activities = network.get_neuron_activities()
        epoch_activity.append(np.mean(activities > 0.5))  # Firing rate

    # Homeostatic scaling applied automatically
    mean_activity = np.mean(epoch_activity)
    activity_history.append(mean_activity)

    print(f"Epoch {epoch}: Mean activity = {mean_activity:.3f}")

# Plot activity stabilization
plt.figure(figsize=(8, 5))
plt.plot(activity_history)
plt.axhline(y=0.1, color='r', linestyle='--', label='Target activity')
plt.xlabel('Epoch')
plt.ylabel('Mean Firing Rate')
plt.title('Homeostatic Activity Regulation')
plt.legend()
plt.grid(True)
plt.show()
```

### Intrinsic Plasticity

```python
# Intrinsic plasticity (neuron-level homeostatic regulation)
intrinsic_plasticity = ln.IntrinsicPlasticity(
    target_probability=0.1,   # Target firing probability
    adaptation_rate=0.0001,   # Adaptation speed
    voltage_dependence=True   # Voltage-dependent adaptation
)

network = ln.Network(neurons=200)

# Apply intrinsic plasticity to neurons
for neuron in network.neurons:
    neuron.set_intrinsic_plasticity(intrinsic_plasticity)

# Training
firing_stats = []

for epoch in range(50):
    firing_counts = np.zeros(200)

    # Present random inputs
    for trial in range(100):
        inputs = np.random.randn(200) * 2.0
        network.set_inputs(inputs)
        network.step(dt=0.01)

        # Count spikes
        spikes = network.get_spikes()
        for spike in spikes:
            firing_counts[spike.neuron_id] += 1

    # Convert to firing rates
    firing_rates = firing_counts / 100
    firing_stats.append(firing_rates)

    print(f"Epoch {epoch}: Mean firing rate = {np.mean(firing_rates):.3f}")

# Plot distribution evolution
plt.figure(figsize=(10, 6))
firing_stats = np.array(firing_stats)

plt.subplot(1, 2, 1)
plt.hist(firing_stats[0], bins=20, alpha=0.7, label='Initial')
plt.hist(firing_stats[-1], bins=20, alpha=0.7, label='Final')
plt.xlabel('Firing Rate')
plt.ylabel('Neuron Count')
plt.title('Firing Rate Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.mean(firing_stats, axis=1))
plt.axhline(y=0.1, color='r', linestyle='--', label='Target')
plt.xlabel('Epoch')
plt.ylabel('Mean Firing Rate')
plt.title('Population Activity')
plt.legend()

plt.tight_layout()
plt.show()
```

## Meta-Plasticity

### Plasticity of Plasticity

```python
# Meta-plasticity based on neuromodulators
meta_plasticity = ln.MetaPlasticity(
    calcium_dependent=True,    # Calcium-modulated learning rate
    neuromodulator_sensitivity={
        'dopamine': 1.5,       # Dopamine enhances plasticity
        'acetylcholine': 0.8,  # ACh modulates attention
        'serotonin': -0.3      # Serotonin can suppress plasticity
    },
    activity_dependence=True   # Activity-dependent modulation
)

# Create network with meta-plasticity
network = ln.Network(neurons=300, connectivity=0.1)
network.set_meta_plasticity(meta_plasticity)

# Add neuromodulators
network.add_neuromodulator(ln.DopamineSystem())
network.add_neuromodulator(ln.AcetylcholineSystem())

# Learning with varying neuromodulator states
learning_rates = []

for epoch in range(100):
    # Modulate neuromodulators based on performance
    performance = np.random.random()  # Simulated performance

    if performance > 0.7:
        network.set_neuromodulator_level('dopamine', 0.8)
        network.set_neuromodulator_level('acetylcholine', 0.6)
    else:
        network.set_neuromodulator_level('dopamine', 0.2)
        network.set_neuromodulator_level('acetylcholine', 0.2)

    # Train with random patterns
    for _ in range(10):
        pattern = np.random.randn(300)
        network.set_inputs(pattern)
        network.step(dt=0.01)

    # Record effective learning rate
    current_lr = network.get_effective_learning_rate()
    learning_rates.append(current_lr)

plt.figure(figsize=(8, 5))
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Effective Learning Rate')
plt.title('Meta-Plasticity Modulation')
plt.grid(True)
plt.show()
```

## Structural Plasticity

### Synaptic Pruning and Formation

```python
# Structural plasticity
structural = ln.StructuralPlasticity(
    pruning_threshold=0.01,     # Prune weak synapses
    formation_rate=0.001,       # Rate of new synapse formation
    max_synapses_per_neuron=100, # Connection limit
    distance_dependent=True,    # Prefer local connections
    activity_dependent=True     # Activity drives rewiring
)

network = ln.Network(neurons=200, initial_connectivity=0.05)
network.set_structural_plasticity(structural)

# Structural development
synapse_counts = []

for epoch in range(200):
    # Normal activity-driven plasticity
    for _ in range(50):
        # Correlated input patterns
        pattern1 = np.random.randn(200)
        pattern2 = pattern1 + 0.5 * np.random.randn(200)  # Correlated

        network.set_inputs(pattern1)
        network.step(dt=0.01)

        network.set_inputs(pattern2)
        network.step(dt=0.01)

    # Structural changes (pruning and formation)
    network.apply_structural_changes()

    # Record connectivity
    weights = network.get_weights()
    active_synapses = np.sum(weights > 0.01)
    synapse_counts.append(active_synapses)

plt.figure(figsize=(8, 5))
plt.plot(synapse_counts)
plt.xlabel('Epoch')
plt.ylabel('Active Synapses')
plt.title('Structural Plasticity Development')
plt.grid(True)
plt.show()

print(f"Final synapse count: {synapse_counts[-1]}")
print(f"Development: {synapse_counts[-1] - synapse_counts[0]} synapses")