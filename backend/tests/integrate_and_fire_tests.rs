#[cfg(test)]
mod tests {
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError,
        neuron::{
            integrate_and_fire::{
                AdaptiveExpLeakyIntegrateAndFireNeuron, AdaptiveLeakyIntegrateAndFireNeuron,
                LeakyIntegrateAndFireNeuron, QuadraticIntegrateAndFireNeuron,
                SimpleLeakyIntegrateAndFire, run_static_input_integrate_and_fire,
            },
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor, GaussianParameters,
                IonotropicNeurotransmitterType, IonotropicType, AMPAReceptor,
            },
        },
    };

    #[test]
    fn test_leaky_integrate_and_fire_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();

        // Test with constant input
        for _ in 0..100 {
            let spiked = neuron.iterate_and_spike(10.0);
            assert!(neuron.current_voltage >= neuron.v_reset);
            assert!(neuron.current_voltage <= neuron.v_th * 2.0); // Should not exceed reasonable bounds
        }

        // Test spiking
        let mut spiked_count = 0;
        for _ in 0..1000 {
            if neuron.iterate_and_spike(50.0) {
                spiked_count += 1;
            }
        }
        assert!(spiked_count > 0, "Neuron should spike with high input");

        Ok(())
    }

    #[test]
    fn test_quadratic_integrate_and_fire_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        // Test with constant input
        for _ in 0..100 {
            let spiked = neuron.iterate_and_spike(10.0);
            assert!(neuron.current_voltage >= neuron.v_reset);
            assert!(neuron.current_voltage <= neuron.v_th * 2.0);
        }

        // Test spiking
        let mut spiked_count = 0;
        for _ in 0..1000 {
            if neuron.iterate_and_spike(50.0) {
                spiked_count += 1;
            }
        }
        assert!(spiked_count > 0, "Neuron should spike with high input");

        Ok(())
    }

    #[test]
    fn test_adaptive_leaky_integrate_and_fire_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = AdaptiveLeakyIntegrateAndFireNeuron::default_impl();

        // Test with constant input
        for _ in 0..100 {
            let spiked = neuron.iterate_and_spike(10.0);
            assert!(neuron.current_voltage >= neuron.v_reset);
            assert!(neuron.current_voltage <= neuron.v_th * 2.0);
            assert!(neuron.w_value >= 0.0); // w should be non-negative
        }

        // Test spiking
        let mut spiked_count = 0;
        for _ in 0..1000 {
            if neuron.iterate_and_spike(50.0) {
                spiked_count += 1;
            }
        }
        assert!(spiked_count > 0, "Neuron should spike with high input");

        Ok(())
    }

    #[test]
    fn test_adaptive_exp_leaky_integrate_and_fire_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = AdaptiveExpLeakyIntegrateAndFireNeuron::default_impl();

        // Test with constant input
        for _ in 0..100 {
            let spiked = neuron.iterate_and_spike(10.0);
            assert!(neuron.current_voltage >= neuron.v_reset);
            assert!(neuron.current_voltage <= neuron.v_th * 2.0);
            assert!(neuron.w_value >= 0.0);
        }

        // Test spiking
        let mut spiked_count = 0;
        for _ in 0..1000 {
            if neuron.iterate_and_spike(50.0) {
                spiked_count += 1;
            }
        }
        assert!(spiked_count > 0, "Neuron should spike with high input");

        Ok(())
    }

    #[test]
    fn test_simple_leaky_integrate_and_fire_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = SimpleLeakyIntegrateAndFire::default_impl();

        // Test with constant input
        for _ in 0..100 {
            let spiked = neuron.iterate_and_spike(10.0);
            assert!(neuron.current_voltage >= neuron.v_reset);
            assert!(neuron.current_voltage <= neuron.v_th * 2.0);
        }

        // Test spiking
        let mut spiked_count = 0;
        for _ in 0..1000 {
            if neuron.iterate_and_spike(50.0) {
                spiked_count += 1;
            }
        }
        assert!(spiked_count > 0, "Neuron should spike with high input");

        Ok(())
    }

    #[test]
    fn test_run_static_input_function() {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();
        let voltages = run_static_input_integrate_and_fire(&mut neuron, 10.0, None, 100);

        assert_eq!(voltages.len(), 100);
        for v in voltages {
            assert!(v >= neuron.v_reset);
            assert!(v <= neuron.v_th * 2.0);
        }
    }

    #[test]
    fn test_run_static_input_with_gaussian() {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();
        let params = GaussianParameters { mean: 10.0, std: 1.0 };
        let voltages = run_static_input_integrate_and_fire(&mut neuron, 10.0, Some(params), 100);

        assert_eq!(voltages.len(), 100);
        for v in voltages {
            assert!(v >= neuron.v_reset);
            assert!(v <= neuron.v_th * 2.0);
        }
    }

    #[test]
    fn test_neurotransmission_basic() -> Result<(), SpikingNeuralNetworksError> {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();

        // Add AMPA receptor
        neuron.receptors.insert(
            IonotropicNeurotransmitterType::AMPA,
            IonotropicType::AMPA(AMPAReceptor::default())
        )?;
        neuron.synaptic_neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA,
            ApproximateNeurotransmitter::default()
        );

        // Test normal iteration
        for _ in 0..50 {
            neuron.iterate_and_spike(5.0);
        }

        // Test with neurotransmitter input
        let concentrations = [
            (IonotropicNeurotransmitterType::AMPA, 1.0),
            (IonotropicNeurotransmitterType::NMDA, 0.0),
            (IonotropicNeurotransmitterType::GABA, 0.0),
        ].into();

        for _ in 0..50 {
            neuron.iterate_with_neurotransmitter_and_spike(5.0, &concentrations);
        }

        Ok(())
    }

    #[test]
    fn test_adaptive_behavior() {
        let mut neuron = AdaptiveLeakyIntegrateAndFireNeuron::default_impl();
        let initial_w = neuron.w_value;

        // Apply high input to cause spiking and w increase
        for _ in 0..200 {
            neuron.iterate_and_spike(100.0);
        }

        // w should have increased due to spiking
        assert!(neuron.w_value > initial_w, "Adaptive value should increase after spiking");
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();
        neuron.current_voltage = neuron.v_th + 1.0; // Force spike

        let spiked = neuron.iterate_and_spike(0.0);
        assert!(spiked, "Should spike when above threshold");

        // During refractory period, voltage should be reset
        assert_eq!(neuron.current_voltage, neuron.v_reset, "Voltage should be reset after spiking");

        // Next few iterations should maintain reset voltage
        for _ in 0..5 {
            neuron.iterate_and_spike(0.0);
            assert_eq!(neuron.current_voltage, neuron.v_reset, "Should stay at reset during refractory");
        }
    }
}