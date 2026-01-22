use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spiking_neural_networks::neuron::{
    hodgkin_huxley::HodgkinHuxleyNeuron,
    integrate_and_fire::{
        IzhikevichNeuron,
        QuadraticIntegrateAndFireNeuron,
        LeakyIntegrateAndFireNeuron,
    },
    iterate_and_spike::IterateAndSpike
};
use spiking_neural_networks::graph::AdjacencyMatrix;
use spiking_neural_networks::neuron::plasticity::{
    STDP, BCM, RewardModulatedSTDP, TripletSTDP
};
use spiking_neural_networks::classifiers::{
    STDPClassifier, RSTDPClassifier, LSMClassifier
};
use spiking_neural_networks::digital_twin::DigitalTwin;

fn bench_neuron_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_models");

    // Izhikevich neuron
    group.bench_function("izhikevich", |b| {
        let mut neuron = IzhikevichNeuron::default_impl();
        b.iter(|| {
            black_box(neuron.iterate_and_spike(30.0));
        });
    });

    // Hodgkin-Huxley neuron
    group.bench_function("hodgkin_huxley", |b| {
        let mut neuron = HodgkinHuxleyNeuron::default_impl();
        b.iter(|| {
            black_box(neuron.iterate_and_spike(30.0));
        });
    });

    // Quadratic Integrate-and-Fire
    group.bench_function("quadratic_if", |b| {
        let mut neuron = QuadraticIntegrateAndFireNeuron::default_impl();
        b.iter(|| {
            black_box(neuron.iterate_and_spike(30.0));
        });
    });

    // Leaky Integrate-and-Fire
    group.bench_function("leaky_if", |b| {
        let mut neuron = LeakyIntegrateAndFireNeuron::default_impl();
        b.iter(|| {
            black_box(neuron.iterate_and_spike(30.0));
        });
    });

    group.finish();
}

fn bench_plasticity(c: &mut Criterion) {
    let mut group = c.benchmark_group("plasticity");

    group.bench_function("stdp", |b| {
        let stdp = STDP::default();
        let pre_time = 1000.0;
        let post_time = 1020.0;
        b.iter(|| {
            black_box(stdp.update_weight(1.0, pre_time, post_time));
        });
    });

    group.bench_function("bcm", |b| {
        let bcm = BCM::default();
        let post_rate = 10.0;
        let avg_post_rate = 8.0;
        b.iter(|| {
            black_box(bcm.update_weight(post_rate, avg_post_rate));
        });
    });

    group.bench_function("reward_modulated_rstdp", |b| {
        let rstdp = RewardModulatedSTDP::default();
        let pre_time = 1000.0;
        let post_time = 1020.0;
        let reward = 1.0;
        b.iter(|| {
            black_box(rstdp.update_weight(1.0, pre_time, post_time, reward));
        });
    });

    group.bench_function("triplet_stdp", |b| {
        let triplet = TripletSTDP::default();
        let pre_time = 1000.0;
        let post_time = 1020.0;
        b.iter(|| {
            black_box(triplet.update_weight(1.0, pre_time, post_time));
        });
    });

    group.finish();
}

fn bench_network_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_operations");

    // Create a small adjacency matrix for testing
    let mut network: AdjacencyMatrix<(usize, usize), f32> = AdjacencyMatrix::default();
    network.add_node((0, 0));
    network.add_node((1, 0));
    network.add_node((2, 0));
    network.edit_weight(&((0, 0)), &((1, 0)), Some(0.5)).unwrap();
    network.edit_weight(&((1, 0)), &((2, 0)), Some(0.8)).unwrap();

    group.bench_function("adjacency_matrix_lookup", |b| {
        b.iter(|| {
            black_box(network.lookup_weight(&((0, 0)), &((1, 0))));
        });
    });

    group.bench_function("adjacency_matrix_edit", |b| {
        b.iter(|| {
            black_box(network.edit_weight(&((0, 0)), &((2, 0)), Some(0.3)));
        });
    });

    group.finish();
}

fn bench_classifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("classifiers");

    // STDP Classifier
    group.bench_function("stdp_classifier", |b| {
        let classifier = STDPClassifier::new(10, 5); // 10 input, 5 output
        let input_pattern = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        b.iter(|| {
            black_box(classifier.predict(&input_pattern));
        });
    });

    // R-STDP Classifier
    group.bench_function("rstdp_classifier", |b| {
        let classifier = RSTDPClassifier::new(10, 5);
        let input_pattern = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        b.iter(|| {
            black_box(classifier.predict(&input_pattern));
        });
    });

    // LSM Classifier
    group.bench_function("lsm_classifier", |b| {
        let classifier = LSMClassifier::new(10, 20, 5); // input, reservoir, output
        let input_pattern = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        b.iter(|| {
            black_box(classifier.predict(&input_pattern));
        });
    });

    group.finish();
}

fn bench_digital_twin(c: &mut Criterion) {
    let mut group = c.benchmark_group("digital_twin");

    // Create a basic digital twin for benchmarking
    let mut digital_twin = DigitalTwin::new();

    group.bench_function("digital_twin_iterate", |b| {
        b.iter(|| {
            black_box(digital_twin.iterate()); // iterate all regions
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_neuron_models,
    bench_plasticity,
    bench_network_operations,
    bench_classifiers,
    bench_digital_twin
);
criterion_main!(benches);