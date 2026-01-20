# Advanced Tutorial: Custom Neuron Models

This tutorial covers creating custom neuron models by implementing the framework's traits and using the provided macros. Custom models allow you to implement novel neural dynamics, integrate domain-specific requirements, or optimize for specific hardware.

## Introduction to Custom Models

The framework provides a trait-based architecture for implementing custom neuron models. You can implement `IterateAndSpike`, `NeurotransmitterKinetics`, and `ReceptorKinetics` traits to create fully customized neurons.

## Basic Custom Neuron Structure

### Implementing the IterateAndSpike Trait

The `IterateAndSpike` trait defines the core neural dynamics.

```rust
use shnn_core::traits::{IterateAndSpike, Spike};
use shnn_core::types::{NeuronState, NeuronParameters};

#[derive(Clone, Debug)]
pub struct CustomNeuron {
    pub state: CustomNeuronState,
    pub params: CustomNeuronParameters,
}

#[derive(Clone, Debug)]
pub struct CustomNeuronState {
    pub voltage: f32,
    pub adaptation: f32,
    pub custom_variable: f32,  // Your custom state variable
}

#[derive(Clone, Debug)]
pub struct CustomNeuronParameters {
    pub v_rest: f32,
    pub v_th: f32,
    pub tau_m: f32,
    pub custom_param: f32,  // Your custom parameter
}

impl IterateAndSpike for CustomNeuron {
    fn iterate_and_spike(&mut self, dt: f32, input_current: f32) -> Option<Spike> {
        // Update voltage using exponential Euler method
        let dv = (self.params.v_rest - self.state.voltage + input_current) / self.params.tau_m;
        self.state.voltage += dv * dt;

        // Update adaptation
        let da = -self.state.adaptation / 100.0;  // Adaptation time constant
        self.state.adaptation += da * dt;

        // Update custom variable
        let d_custom = (self.params.custom_param - self.state.custom_variable) / 50.0;
        self.state.custom_variable += d_custom * dt;

        // Spike condition with adaptation
        if self.state.voltage > self.params.v_th + self.state.adaptation {
            // Reset after spike
            self.state.voltage = self.params.v_rest;
            self.state.adaptation += 2.0;  // Increase adaptation

            Some(Spike {
                timestamp: 0,  // Will be set by lattice
                neuron_id: 0,  // Will be set by lattice
                strength: 1.0,
            })
        } else {
            None
        }
    }

    fn get_voltage(&self) -> f32 {
        self.state.voltage
    }

    fn set_voltage(&mut self, voltage: f32) {
        self.state.voltage = voltage;
    }

    fn get_parameters(&self) -> &NeuronParameters {
        // Convert to framework parameters if needed
        todo!()
    }
}
```

### Using the Creation Macro

The framework provides a macro to easily integrate custom neurons into lattices.

```rust
use shnn_core::macros::raw_create_agent_type_for_lattice;

// Create the lattice agent type
raw_create_agent_type_for_lattice!(
    CustomNeuron,
    CustomNeuronState,
    CustomNeuronParameters
);

// Now you can use CustomNeuron in lattices
let mut lattice = IzhikevichLattice::new();
lattice.populate(CustomNeuron::new(), 10, 10);
```

## Neurotransmitter Kinetics

### Implementing NeurotransmitterKinetics

For neurons that release neurotransmitters:

```rust
use shnn_core::traits::NeurotransmitterKinetics;
use shnn_core::neurotransmitters::{NeurotransmitterType, NeurotransmitterConcentration};

impl NeurotransmitterKinetics for CustomNeuron {
    fn get_neurotransmitter_type(&self) -> NeurotransmitterType {
        NeurotransmitterType::Glutamate  // Or your custom type
    }

    fn calculate_release_probability(&self, stimulation: f32) -> f32 {
        // Calcium-dependent release
        let calcium = self.state.custom_variable;  // Using custom variable as [Ca²⁺]
        0.1 * (1.0 + (calcium / 0.5).tanh())  // Sigmoid calcium dependence
    }

    fn update_neurotransmitter_concentration(
        &mut self,
        dt: f32,
        released_amount: f32,
        reuptake_rate: f32
    ) {
        // Simple exponential decay with release
        let decay = -self.state.neurotransmitter_conc * reuptake_rate;
        let release = released_amount;
        self.state.neurotransmitter_conc += (decay + release) * dt;

        // Ensure non-negative
        self.state.neurotransmitter_conc = self.state.neurotransmitter_conc.max(0.0);
    }

    fn get_neurotransmitter_concentration(&self) -> f32 {
        self.state.neurotransmitter_conc
    }
}
```

### Custom Neurotransmitter Types

```rust
#[derive(Clone, Debug)]
pub enum CustomNeurotransmitterType {
    Standard(NeurotransmitterType),
    Novel(CustomNovelNT),
}

#[derive(Clone, Debug)]
pub struct CustomNovelNT {
    pub name: String,
    pub diffusion_rate: f32,
    pub receptor_binding: f32,
}
```

## Receptor Kinetics

### Implementing ReceptorKinetics

For neurons with receptor dynamics:

```rust
use shnn_core::traits::ReceptorKinetics;
use shnn_core::receptors::{ReceptorState, ReceptorParameters};

impl ReceptorKinetics for CustomNeuron {
    fn update_receptor_state(&mut self, dt: f32, neurotransmitter_conc: f32) {
        // Simple two-state receptor model
        let binding_rate = 0.1 * neurotransmitter_conc;
        let unbinding_rate = 0.05;

        // Update bound fraction
        let d_bound = binding_rate * (1.0 - self.state.receptor_bound) -
                     unbinding_rate * self.state.receptor_bound;
        self.state.receptor_bound += d_bound * dt;

        // Clamp to [0, 1]
        self.state.receptor_bound = self.state.receptor_bound.clamp(0.0, 1.0);
    }

    fn get_receptor_current(&self, reversal_potential: f32) -> f32 {
        // Current through open channels
        let g_max = 10.0;  // Maximum conductance
        let conductance = g_max * self.state.receptor_bound * self.state.open_probability;

        conductance * (reversal_potential - self.state.voltage)
    }

    fn get_receptor_state(&self) -> &ReceptorState {
        &self.state.receptor_state
    }
}
```

## Advanced Custom Models

### Hodgkin-Huxley Style Neuron

```rust
#[derive(Clone, Debug)]
pub struct HHNeuron {
    pub state: HHNeuronState,
    pub params: HHNeuronParameters,
}

#[derive(Clone, Debug)]
pub struct HHNeuronState {
    pub v: f32,        // Membrane potential
    pub m: f32,        // Na+ activation
    pub h: f32,        // Na+ inactivation
    pub n: f32,        // K+ activation
}

impl IterateAndSpike for HHNeuron {
    fn iterate_and_spike(&mut self, dt: f32, i_ext: f32) -> Option<Spike> {
        // Hodgkin-Huxley equations
        let v = self.state.v;

        // Sodium current
        let g_na = 120.0; let e_na = 115.0;
        let i_na = g_na * self.state.m.powi(3) * self.state.h * (v - e_na);

        // Potassium current
        let g_k = 36.0; let e_k = -12.0;
        let i_k = g_k * self.state.n.powi(4) * (v - e_k);

        // Leak current
        let g_l = 0.3; let e_l = 10.607;
        let i_l = g_l * (v - e_l);

        // Membrane equation
        let c_m = 1.0;  // Membrane capacitance
        let dv = (i_ext - i_na - i_k - i_l) / c_m;
        self.state.v += dv * dt;

        // Update gating variables
        self.update_gates(dt);

        // Spike detection
        if self.state.v > 0.0 && self.prev_v <= 0.0 {
            Some(Spike::new())
        } else {
            None
        }
    }
}

impl HHNeuron {
    fn update_gates(&mut self, dt: f32) {
        let v = self.state.v;

        // Alpha and beta functions for m
        let alpha_m = 0.1 * (v + 40.0) / (1.0 - (-(v + 40.0)/10.0).exp());
        let beta_m = 4.0 * (-(v + 65.0)/18.0).exp();
        let tau_m = 1.0 / (alpha_m + beta_m);
        let m_inf = alpha_m * tau_m;
        self.state.m += (m_inf - self.state.m) / tau_m * dt;

        // Similar for h and n...
        // (Implementation omitted for brevity)
    }
}
```

### Adaptive Resonance Theory (ART) Neuron

```rust
#[derive(Clone, Debug)]
pub struct ARTNeuron {
    pub state: ARTNeuronState,
    pub params: ARTNeuronParameters,
}

impl IterateAndSpike for ARTNeuron {
    fn iterate_and_spike(&mut self, dt: f32, input_pattern: f32) -> Option<Spike> {
        // ART-style vigilance and resonance
        let vigilance = self.params.vigilance;
        let input_similarity = self.calculate_similarity(input_pattern);

        if input_similarity > vigilance {
            // Resonance - strengthen memory
            self.state.memory_trace += input_similarity * dt;
            self.state.resonance_level += dt;

            // Check if resonance threshold reached
            if self.state.resonance_level > self.params.resonance_threshold {
                return Some(Spike::new());
            }
        } else {
            // Reset - search for new category
            self.state.resonance_level = 0.0;
            self.state.search_mode = true;
        }

        None
    }
}
```

## GPU-Accelerated Custom Models

### CUDA Kernel Implementation

```rust
// Custom CUDA kernel for your neuron model
const CUSTOM_NEURON_KERNEL: &str = r#"
__global__ void custom_neuron_update(
    float* voltages,
    float* adaptations,
    float* custom_vars,
    const float* params,
    const float* inputs,
    int n_neurons,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    float v = voltages[idx];
    float a = adaptations[idx];
    float c = custom_vars[idx];

    // Your custom update logic here
    float dv = (params[0] - v + inputs[idx]) / params[1];
    v += dv * dt;

    float da = -a / params[2];
    a += da * dt;

    float dc = (params[3] - c) / params[4];
    c += dc * dt;

    // Spike condition
    if (v > params[5] + a) {
        v = params[0];  // Reset
        a += params[6]; // Adaptation increase
        // Record spike...
    }

    voltages[idx] = v;
    adaptations[idx] = a;
    custom_vars[idx] = c;
}
"#;

// Load into GPU lattice
cuda_lattice.load_custom_kernel("custom_update", CUSTOM_NEURON_KERNEL);
cuda_lattice.use_custom_update("custom_update");
```

## Python Interface for Custom Models

### Exposing Custom Models to Python

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct PyCustomNeuron {
    inner: CustomNeuron,
}

#[pymethods]
impl PyCustomNeuron {
    #[new]
    fn new(v_rest: f32, v_th: f32, tau_m: f32, custom_param: f32) -> Self {
        let params = CustomNeuronParameters {
            v_rest,
            v_th,
            tau_m,
            custom_param,
        };
        let state = CustomNeuronState {
            voltage: v_rest,
            adaptation: 0.0,
            custom_variable: 0.0,
        };
        Self {
            inner: CustomNeuron { state, params },
        }
    }

    fn iterate_and_spike(&mut self, dt: f32, input_current: f32) -> Option<PySpike> {
        self.inner.iterate_and_spike(dt, input_current)
            .map(|spike| PySpike::from(spike))
    }
}
```

## Testing Custom Models

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_neuron_basic() {
        let mut neuron = CustomNeuron::new();

        // Test no spike at resting potential
        assert!(neuron.iterate_and_spike(1.0, 0.0).is_none());

        // Test spike with strong input
        assert!(neuron.iterate_and_spike(1.0, 100.0).is_some());
    }

    #[test]
    fn test_neurotransmitter_kinetics() {
        let mut neuron = CustomNeuron::new();

        // Test release probability increases with stimulation
        let prob1 = neuron.calculate_release_probability(0.0);
        let prob2 = neuron.calculate_release_probability(1.0);
        assert!(prob2 > prob1);
    }
}
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_custom_neuron(c: &mut Criterion) {
    let mut neuron = CustomNeuron::new();

    c.bench_function("custom_neuron_update", |b| {
        b.iter(|| {
            black_box(neuron.iterate_and_spike(0.1, black_box(5.0)));
        })
    });
}

criterion_group!(benches, benchmark_custom_neuron);
criterion_main!(benches);
```

## Performance Optimization

### Memory Layout Optimization

```rust
// Structure of arrays (SoA) for better cache performance
#[derive(Clone, Debug)]
pub struct CustomNeuronSoA {
    pub voltages: Vec<f32>,
    pub adaptations: Vec<f32>,
    pub custom_vars: Vec<f32>,
    pub params: CustomNeuronParameters,
}

impl IterateAndSpike for CustomNeuronSoA {
    fn iterate_and_spike(&mut self, dt: f32, input_currents: &[f32]) -> Vec<Option<Spike>> {
        let n = self.voltages.len();
        let mut spikes = Vec::with_capacity(n);

        for i in 0..n {
            // Update all neurons in parallel-friendly manner
            let v = &mut self.voltages[i];
            let a = &mut self.adaptations[i];
            let c = &mut self.custom_vars[i];
            let i_ext = input_currents[i];

            // Update equations...
            // (implementation optimized for SIMD/vectorization)
        }

        spikes
    }
}
```

### SIMD Vectorization

```rust
use std::simd::{f32x4, SimdFloat};

impl CustomNeuronSoA {
    fn update_simd(&mut self, dt: f32, inputs: &[f32]) {
        let chunk_size = 4;  // SIMD width

        for chunk in self.voltages.chunks_exact_mut(chunk_size) {
            let v = f32x4::from_slice(chunk);
            let a = f32x4::from_slice(&self.adaptations[chunk.as_ptr() as usize..]);
            let c = f32x4::from_slice(&self.custom_vars[chunk.as_ptr() as usize..]);
            let i_ext = f32x4::from_slice(&inputs[chunk.as_ptr() as usize..]);

            // Vectorized update
            let dv = (f32x4::splat(self.params.v_rest) - v + i_ext) /
                     f32x4::splat(self.params.tau_m);
            let new_v = v + dv * f32x4::splat(dt);

            // Store results
            new_v.copy_to_slice(chunk);
        }
    }
}
```

## Best Practices

### Design Guidelines

1. **Start Simple**: Begin with basic `IterateAndSpike` implementation
2. **Test Incrementally**: Add features one at a time with tests
3. **Profile Performance**: Use benchmarks to identify bottlenecks
4. **Document Parameters**: Clearly document all parameters and their units
5. **Validate Ranges**: Add bounds checking for parameters

### Common Patterns

- **State Validation**: Check state variables remain in valid ranges
- **Parameter Bounds**: Validate parameters at creation time
- **Numerical Stability**: Use appropriate integration methods
- **Memory Efficiency**: Prefer stack allocation for small structures

### Error Handling

```rust
#[derive(Debug)]
pub enum CustomNeuronError {
    InvalidParameter(String),
    NumericalInstability,
    OutOfBounds,
}

impl CustomNeuron {
    pub fn new(params: CustomNeuronParameters) -> Result<Self, CustomNeuronError> {
        // Validate parameters
        if params.tau_m <= 0.0 {
            return Err(CustomNeuronError::InvalidParameter(
                "tau_m must be positive".to_string()
            ));
        }

        Ok(Self {
            state: CustomNeuronState::default(),
            params,
        })
    }
}
```

## Next Steps

- Apply custom models to [GPU Acceleration](gpu-usage.md)
- Explore plasticity with custom neurons
- See examples using custom models in the [Examples Gallery](../examples/gallery.md)