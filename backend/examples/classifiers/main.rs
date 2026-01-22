use std::fs::File;
use std::io::{BufWriter, Write};
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    classifiers::{Classifier, Regressor, STDPClassifier, RSTDPClassifier, LSMClassifier, RSTDPRegressor, metrics},
};

/// Example usage of classifiers and regressors
fn main() -> Result<(), SpikingNeuralNetworksError> {
    println!("Training STDP-based unsupervised classifier...");

    // Sample data: simple patterns
    let train_inputs = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
    ];
    let train_labels = vec![0, 1, 2, 0, 1]; // For supervised, but STDP ignores

    // STDP Classifier
    let mut stdp_classifier = STDPClassifier::new(3, 3);
    stdp_classifier.train(&train_inputs, &train_labels)?;

    let test_inputs = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let stdp_predictions: Vec<usize> = test_inputs.iter().map(|inp| stdp_classifier.predict(inp)).collect();
    let stdp_accuracy = metrics::accuracy(&stdp_predictions, &[0, 1, 2]);
    println!("STDP Classifier Accuracy: {:.2}", stdp_accuracy);

    // R-STDP Classifier
    println!("Training R-STDP classifier...");
    let mut rstdp_classifier = RSTDPClassifier::new(3, 3);
    rstdp_classifier.train(&train_inputs, &train_labels)?;

    let rstdp_predictions: Vec<usize> = test_inputs.iter().map(|inp| rstdp_classifier.predict(inp)).collect();
    let rstdp_accuracy = metrics::accuracy(&rstdp_predictions, &[0, 1, 2]);
    println!("R-STDP Classifier Accuracy: {:.2}", rstdp_accuracy);

    // LSM Classifier
    println!("Training LSM classifier...");
    let mut lsm_classifier = LSMClassifier::new(3, 10, 3);
    lsm_classifier.train(&train_inputs, &train_labels)?;

    let lsm_predictions: Vec<usize> = test_inputs.iter().map(|inp| lsm_classifier.predict(inp)).collect();
    let lsm_accuracy = metrics::accuracy(&lsm_predictions, &[0, 1, 2]);
    println!("LSM Classifier Accuracy: {:.2}", lsm_accuracy);

    // R-STDP Regressor
    println!("Training R-STDP regressor...");
    let train_targets = vec![1.0, 2.0, 3.0, 1.5, 2.5];
    let mut regressor = RSTDPRegressor::new(3);
    regressor.train(&train_inputs, &train_targets)?;

    let test_targets = vec![1.0, 2.0, 3.0];
    let predictions: Vec<f32> = test_inputs.iter().map(|inp| regressor.predict(inp)).collect();
    let mse = metrics::mse(&predictions, &test_targets);
    println!("R-STDP Regressor MSE: {:.4}", mse);

    // Write results to file
    let mut file = BufWriter::new(File::create("classifier_results.txt")?);
    writeln!(file, "STDP Accuracy: {:.2}", stdp_accuracy)?;
    writeln!(file, "R-STDP Accuracy: {:.2}", rstdp_accuracy)?;
    writeln!(file, "LSM Accuracy: {:.2}", lsm_accuracy)?;
    writeln!(file, "Regressor MSE: {:.4}", mse)?;
    writeln!(file, "Predictions: {:?}", predictions)?;

    println!("Results written to classifier_results.txt");

    Ok(())
}