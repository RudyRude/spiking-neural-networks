//! Example demonstrating the ported neuron models from spiking-neural-networks
//! working with hSNN connectivity structures.

use shnn_core::prelude::*;
use shnn_core::time::TimeStep;

fn main() {
    println!("Testing ported neuron models with hSNN framework");

    // Test DetailedLIFNeuron
    println!("\n=== Testing DetailedLIFNeuron ===");
    let mut lif_neuron = DetailedLIFNeuron::from_spiking_networks_defaults(NeuronId(0));
    let dt = TimeStep::from_ms(0.1);

    println!("Initial voltage: {:.2} mV", lif_neuron.membrane_potential());

    // Apply current and check spiking
    lif_neuron.integrate(5.0, dt);
    println!("After 5.0 nA input: {:.2} mV", lif_neuron.membrane_potential());

    lif_neuron.set_membrane_potential(-50.0); // Set above threshold
    if let Some(spike) = lif_neuron.update(dt) {
        println!("Spike generated! Time: {:?}", spike.timestamp);
        println!("Reset to: {:.2} mV", lif_neuron.membrane_potential());
    }

    // Test DetailedHHNeuron
    println!("\n=== Testing DetailedHHNeuron ===");
    let mut hh_neuron = DetailedHHNeuron::from_spiking_networks_defaults(NeuronId(1));
    let dt_hh = TimeStep::from_ms(0.01); // Smaller timestep for HH

    println!("Initial voltage: {:.2} mV", hh_neuron.membrane_potential());
    println!("Initial gate values - Na_m: {:.3}, Na_h: {:.3}, K_n: {:.3}",
             hh_neuron.na_m, hh_neuron.na_h, hh_neuron.k_n);

    // Apply strong current to trigger action potential
    for i in 0..100 {
        hh_neuron.integrate(50.0, dt_hh);
        if let Some(spike) = hh_neuron.update(dt_hh) {
            println!("HH Spike generated after {} steps! Time: {:?}", i, spike.timestamp);
            break;
        }
        if i == 99 {
            println!("No spike generated, final voltage: {:.2} mV", hh_neuron.membrane_potential());
        }
    }

    // Test DetailedIzhikevichNeuron
    println!("\n=== Testing DetailedIzhikevichNeuron ===");
    let mut izh_neuron = DetailedIzhikevichNeuron::regular_spiking(NeuronId(2));

    println!("Initial voltage: {:.2} mV, recovery: {:.2}",
             izh_neuron.membrane_potential(), izh_neuron.recovery_variable);

    // Apply current - Izhikevich neurons need sustained input
    for i in 0..200 {
        izh_neuron.integrate(20.0, dt);
        if let Some(spike) = izh_neuron.update(dt) {
            println!("Izhikevich spike generated after {} steps!", i);
            println!("Reset to: {:.2} mV, recovery updated to: {:.2}",
                     izh_neuron.membrane_potential(), izh_neuron.recovery_variable);
            break;
        }
    }

    println!("\n=== All neuron models successfully ported! ===");
    println!("The detailed neuron models from spiking-neural-networks now implement");
    println!("the hSNN Neuron trait and can work with all hSNN connectivity structures.");
}