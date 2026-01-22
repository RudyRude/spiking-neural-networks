#[cfg(test)]
mod plasticity_tests {
    use spiking_neural_networks::{
        neuron::{
            integrate_and_fire::IzhikevichNeuron,
            iterate_and_spike::{IterateAndSpike, LastFiringTime},
            plasticity::{
                Plasticity, STDP, BCM, RewardModulatedSTDP, TraceRSTDP, TripletSTDP, TripletWeight,
                BCMActivity,
            },
        },
        error::SpikingNeuralNetworksError,
    };

    /// Test STDP weight update for pre-before-post (LTP)
    #[test]
    fn test_stdp_ltp() {
        let stdp = STDP::default();
        let mut weight = 1.0;

        // Create mock neurons with firing times
        let pre_neuron = MockNeuron { last_firing_time: Some(10) };
        let post_neuron = MockNeuron { last_firing_time: Some(15) };

        stdp.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Should increase weight (LTP)
        assert!(weight > 1.0);
    }

    /// Test STDP weight update for post-before-pre (LTD)
    #[test]
    fn test_stdp_ltd() {
        let stdp = STDP::default();
        let mut weight = 1.0;

        let pre_neuron = MockNeuron { last_firing_time: Some(15) };
        let post_neuron = MockNeuron { last_firing_time: Some(10) };

        stdp.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Should decrease weight (LTD)
        assert!(weight < 1.0);
    }

    /// Test STDP do_update returns true when neuron is spiking
    #[test]
    fn test_stdp_do_update_spiking() {
        let stdp = STDP::default();
        let spiking_neuron = MockSpikingNeuron { is_spiking: true };
        assert!(stdp.do_update(&spiking_neuron));
    }

    /// Test STDP do_update returns false when neuron is not spiking
    #[test]
    fn test_stdp_do_update_not_spiking() {
        let stdp = STDP::default();
        let non_spiking_neuron = MockSpikingNeuron { is_spiking: false };
        assert!(!stdp.do_update(&non_spiking_neuron));
    }

    /// Test BCM plasticity
    #[test]
    fn test_bcm_plasticity() {
        let bcm = BCM::default();
        let mut weight = 1.0;

        let pre_neuron = MockBCMNeuron { activity: 0.8 };
        let post_neuron = MockBCMNeuron { activity: 0.6 };

        bcm.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Weight should change based on BCM rule
        assert_ne!(weight, 1.0);
    }

    /// Test BCM do_update
    #[test]
    fn test_bcm_do_update() {
        let bcm = BCM::default();
        let neuron = MockSpikingNeuron { is_spiking: true };
        assert!(bcm.do_update(&neuron));
    }

    /// Test Reward Modulated STDP
    #[test]
    fn test_reward_modulated_stdp() {
        let mut rm_stdp = RewardModulatedSTDP::default();
        let mut weight = TraceRSTDP::default();

        let pre_neuron = MockNeuron { last_firing_time: Some(10) };
        let post_neuron = MockNeuron { last_firing_time: Some(15) };

        // Apply reward
        rm_stdp.update(1.0);

        rm_stdp.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Weight should be updated
        assert_ne!(weight.weight, 0.0);
    }

    /// Test Triplet STDP LTP
    #[test]
    fn test_triplet_stdp_ltp() {
        let triplet_stdp = TripletSTDP::default();
        let mut weight = TripletWeight::default();

        let pre_neuron = MockNeuron { last_firing_time: Some(10) };
        let post_neuron = MockNeuron { last_firing_time: Some(15) };

        triplet_stdp.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Should increase weight for LTP
        assert!(weight.weight > 1.0);
    }

    /// Test Triplet STDP LTD
    #[test]
    fn test_triplet_stdp_ltd() {
        let triplet_stdp = TripletSTDP::default();
        let mut weight = TripletWeight::default();

        let pre_neuron = MockNeuron { last_firing_time: Some(15) };
        let post_neuron = MockNeuron { last_firing_time: Some(10) };

        triplet_stdp.update_weight(&mut weight, &pre_neuron, &post_neuron);

        // Should decrease weight for LTD
        assert!(weight.weight < 1.0);
    }

    /// Test Triplet STDP do_update
    #[test]
    fn test_triplet_stdp_do_update() {
        let triplet_stdp = TripletSTDP::default();
        let neuron = MockSpikingNeuron { is_spiking: true };
        assert!(triplet_stdp.do_update(&neuron));
    }

    /// Test TripletWeight default
    #[test]
    fn test_triplet_weight_default() {
        let weight = TripletWeight::default();
        assert_eq!(weight.weight, 1.0);
        assert_eq!(weight.traces.r1, 0.0);
        assert_eq!(weight.traces.r2, 0.0);
        assert_eq!(weight.traces.o1, 0.0);
        assert_eq!(weight.traces.o2, 0.0);
    }

    /// Test TripletTraces default
    #[test]
    fn test_triplet_traces_default() {
        let traces = Default::default();
        assert_eq!(traces.r1, 0.0);
        assert_eq!(traces.r2, 0.0);
        assert_eq!(traces.o1, 0.0);
        assert_eq!(traces.o2, 0.0);
    }

    /// Test plasticity parameters are within valid ranges
    #[test]
    fn test_plasticity_parameter_ranges() {
        let stdp = STDP::default();
        assert!(stdp.a_plus > 0.0);
        assert!(stdp.a_minus > 0.0);
        assert!(stdp.tau_plus > 0.0);
        assert!(stdp.tau_minus > 0.0);
        assert!(stdp.dt > 0.0);

        let triplet = TripletSTDP::default();
        assert!(triplet.a2_plus > 0.0);
        assert!(triplet.a2_minus > 0.0);
        assert!(triplet.a3_plus > 0.0);
        assert!(triplet.a3_minus > 0.0);
        assert!(triplet.tau_plus > 0.0);
        assert!(triplet.tau_minus > 0.0);
        assert!(triplet.tau_x > 0.0);
        assert!(triplet.tau_y > 0.0);
        assert!(triplet.dt > 0.0);
    }

    // Mock structs for testing

    struct MockNeuron {
        last_firing_time: Option<usize>,
    }

    impl LastFiringTime for MockNeuron {
        fn get_last_firing_time(&self) -> Option<usize> {
            self.last_firing_time
        }
    }

    struct MockSpikingNeuron {
        is_spiking: bool,
    }

    impl IterateAndSpike for MockSpikingNeuron {
        type N = spiking_neural_networks::neuron::iterate_and_spike::ApproximateNeurotransmitterType;

        fn iterate_and_spike(&mut self, _input_current: f32) -> bool {
            self.is_spiking
        }

        fn iterate_with_neurotransmitter_and_spike(
            &mut self,
            _input_current: f32,
            _neurotransmitter_concentrations: &spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterConcentrations<Self::N>,
        ) -> bool {
            self.is_spiking
        }

        fn get_neurotransmitter_concentrations(&self) -> spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterConcentrations<Self::N> {
            Default::default()
        }

        fn set_last_firing_time(&mut self, _time: Option<usize>) {}
    }

    impl LastFiringTime for MockSpikingNeuron {
        fn get_last_firing_time(&self) -> Option<usize> {
            None
        }
    }

    struct MockBCMNeuron {
        activity: f32,
    }

    impl LastFiringTime for MockBCMNeuron {
        fn get_last_firing_time(&self) -> Option<usize> {
            None
        }
    }

    impl BCMActivity for MockBCMNeuron {
        fn get_activity(&self) -> f32 {
            self.activity
        }

        fn get_averaged_activity(&self) -> f32 {
            self.activity
        }
    }
}