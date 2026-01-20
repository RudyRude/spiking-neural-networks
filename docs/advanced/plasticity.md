# Advanced Tutorial: Synaptic Plasticity

Synaptic plasticity allows neural networks to learn and adapt by modifying connection strengths based on activity patterns. This tutorial covers the plasticity mechanisms available in the framework.

## Introduction to Plasticity

Plastic changes in synaptic strength are fundamental to learning and memory in biological neural systems. The framework implements several forms of spike-timing-dependent plasticity (STDP) and other learning rules.

## Spike-Timing-Dependent Plasticity (STDP)

STDP modifies synaptic weights based on the relative timing of pre- and postsynaptic spikes.

### Basic STDP

```python
import lixirnet as ln
import numpy as np
import matplotlib.pyplot as plt

# Create lattice with STDP-enabled synapses
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 10, 10)

# Enable STDP with standard parameters
stdp_rule = ln.STDPRule(
    a_plus=0.01,   # LTP amplitude
    a_minus=0.0105, # LTD amplitude
    tau_plus=20.0,  # LTP time constant (ms)
    tau_minus=20.0  # LTD time constant (ms)
)

# Connect with plastic synapses
def distance_connection(pos1, pos2):
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance <= 3 and pos1 != pos2

lattice.connect(distance_connection, plasticity_rule=stdp_rule)
```

### Custom STDP Windows

```python
# Asymmetric STDP (favoring LTP)
asymmetric_stdp = ln.STDPRule(
    a_plus=0.02,    # Strong LTP
    a_minus=0.005,  # Weak LTD
    tau_plus=10.0,  # Fast LTP decay
    tau_minus=30.0  # Slow LTD decay
)

# Symmetric STDP
symmetric_stdp = ln.STDPRule(
    a_plus=0.01,
    a_minus=0.01,
    tau_plus=20.0,
    tau_minus=20.0
)
```

## Homeostatic Plasticity

Homeostatic plasticity maintains network stability by scaling synaptic strengths.

```python
# Homeostatic scaling
homeostatic_rule = ln.HomeostaticRule(
    target_rate=10.0,     # Target firing rate (Hz)
    eta=0.001,           # Learning rate
    time_window=1000     # Integration window (ms)
)

lattice.connect(
    distance_connection,
    plasticity_rule=homeostatic_rule
)
```

## BCM Theory (Bienenstock-Cooper-Munro)

BCM theory provides a sliding threshold for LTP/LTD based on postsynaptic activity.

```python
bcm_rule = ln.BCMRule(
    eta=0.001,          # Learning rate
    theta_0=2.0,        # Initial threshold
    tau_theta=1000.0,   # Threshold adaptation time constant
    alpha=1.0           # Threshold scaling
)

lattice.connect(distance_connection, plasticity_rule=bcm_rule)
```

## Hebbian Learning

Classical Hebbian learning: "neurons that fire together wire together".

```python
hebbian_rule = ln.HebbianRule(
    eta=0.001,      # Learning rate
    threshold=0.5   # Activity threshold
)

lattice.connect(distance_connection, plasticity_rule=hebbian_rule)
```

## Oja's Rule

Oja's rule for principal component analysis and feature extraction.

```python
oja_rule = ln.OjaRule(
    eta=0.001,      # Learning rate
    beta=0.1        # Normalization factor
)

# Best for linear neurons or rate-based models
lattice.connect(distance_connection, plasticity_rule=oja_rule)
```

## Multi-Rule Plasticity

Combine multiple plasticity mechanisms:

```python
# STDP + Homeostasis
combined_rule = ln.CombinedPlasticityRule([
    ln.STDPRule(a_plus=0.005, a_minus=0.0051),
    ln.HomeostaticRule(target_rate=5.0, eta=0.0001)
])

lattice.connect(distance_connection, plasticity_rule=combined_rule)
```

## Running Plasticity Simulations

### Training Protocol

```python
# Setup plastic lattice
lattice = ln.IzhikevichLattice()
lattice.populate(ln.IzhikevichNeuron(), 20, 20)
lattice.connect(distance_connection, plasticity_rule=stdp_rule)

# Record weight evolution
initial_weights = lattice.get_weights().copy()
weight_history = []

# Training phase
for epoch in range(10):
    # Reset activity but keep weights
    lattice.reset_history()
    lattice.reset_timing()

    # Apply input pattern
    apply_input_pattern(lattice, epoch)

    # Run simulation
    lattice.run_lattice(1000)

    # Record weights
    weight_history.append(lattice.get_weights().copy())

    print(f"Epoch {epoch}: {len(lattice.spike_times)} spikes")
```

### Input Pattern Application

```python
def apply_input_pattern(lattice, pattern_id):
    """Apply different input patterns"""
    if pattern_id == 0:
        # Vertical bars
        for y in range(lattice.height):
            lattice.grid[0][y].add_current(5.0)  # Left edge
    elif pattern_id == 1:
        # Horizontal bars
        for x in range(lattice.width):
            lattice.grid[x][0].add_current(5.0)  # Top edge
    # Add more patterns...
```

## Analyzing Plasticity Effects

### Weight Distribution Evolution

```python
def plot_weight_evolution(weight_history):
    """Plot how weight distribution changes over time"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for epoch, weights in enumerate(weight_history):
        ax = axes[epoch // 5, epoch % 5]
        ax.hist(weights.flatten(), bins=50, alpha=0.7)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

plot_weight_evolution(weight_history)
```

### Weight Matrix Visualization

```python
def plot_weight_matrix(lattice):
    """Visualize synaptic weight matrix"""
    weights = lattice.get_weights()
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Synaptic Weight')
    plt.xlabel('Postsynaptic Neuron')
    plt.ylabel('Presynaptic Neuron')
    plt.title('Synaptic Weight Matrix')
    plt.show()

plot_weight_matrix(lattice)
```

### Plasticity Metrics

```python
def analyze_plasticity(lattice, initial_weights):
    """Analyze changes in synaptic structure"""
    final_weights = lattice.get_weights()

    # Weight change distribution
    weight_changes = final_weights - initial_weights
    plt.hist(weight_changes.flatten(), bins=50)
    plt.xlabel('Weight Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Synaptic Changes')
    plt.show()

    # Connection statistics
    strengthened = np.sum(weight_changes > 0.01)
    weakened = np.sum(weight_changes < -0.01)
    unchanged = np.sum(np.abs(weight_changes) <= 0.01)

    print(f"Strengthened synapses: {strengthened}")
    print(f"Weakened synapses: {weakened}")
    print(f"Unchanged synapses: {unchanged}")

analyze_plasticity(lattice, initial_weights)
```

## Advanced Plasticity Features

### Structural Plasticity

```python
# Enable synapse creation/removal
structural_rule = ln.StructuralPlasticityRule(
    creation_threshold=0.8,    # Correlated activity threshold
    removal_threshold=0.1,     # Weak synapse removal
    max_synapses=1000          # Maximum synapses per neuron
)

lattice.connect(distance_connection, plasticity_rule=structural_rule)
```

### Meta-Plasticity

```python
# Plasticity of plasticity parameters
meta_rule = ln.MetaPlasticityRule(
    base_rule=stdp_rule,
    modulation_factor=0.1,
    adaptation_rate=0.001
)

lattice.connect(distance_connection, plasticity_rule=meta_rule)
```

## Performance Considerations

1. **Memory Usage**: Plasticity tracking increases memory requirements
2. **Computation**: Plasticity updates add computational overhead
3. **Numerical Stability**: Small learning rates prevent instability
4. **Convergence**: Monitor weight distributions for convergence

## Applications

- **Pattern Recognition**: STDP for temporal sequence learning
- **Motor Learning**: BCM theory for motor skill acquisition
- **Memory Formation**: Hebbian learning for associative memory
- **Feature Extraction**: Oja's rule for dimensionality reduction
- **Homeostasis**: Maintaining network stability during learning

## Troubleshooting

### Common Issues

- **Weight Explosion**: Reduce learning rates or add weight bounds
- **No Learning**: Check spike timing precision and learning rates
- **Instability**: Add homeostatic mechanisms or weight normalization
- **Slow Convergence**: Increase learning rates or adjust time constants

### Debugging Tips

```python
# Monitor weight bounds
weights = lattice.get_weights()
print(f"Weight range: {weights.min():.3f} to {weights.max():.3f}")

# Check for NaN/inf values
if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
    print("Warning: Invalid weight values detected")

# Analyze spike correlations
from scipy.stats import pearsonr
pre_spikes = get_presynaptic_activity(lattice)
post_spikes = get_postsynaptic_activity(lattice)
correlation, _ = pearsonr(pre_spikes, post_spikes)
print(f"Pre-post correlation: {correlation:.3f}")
```

## Next Steps

- Explore [Neurotransmitter Dynamics](neurotransmitter-dynamics.md) for biochemical signaling
- Learn about [GPU Acceleration](gpu-usage.md) for faster plasticity simulations
- See plasticity examples in the [Examples Gallery](../examples/gallery.md)