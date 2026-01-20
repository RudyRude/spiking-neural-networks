//! Integration tests for the NDFH CLI tool.
//! These tests validate end-to-end CLI functionality.

use std::process::Command;

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "ndfh-cli", "--", "--help"])
        .output()
        .expect("Failed to execute CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ndfh-cli"));
}

#[test]
fn test_cli_version() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "ndfh-cli", "--", "--version"])
        .output()
        .expect("Failed to execute CLI");

    assert!(output.status.success());
}

#[test]
fn test_cli_invalid_command() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "ndfh-cli", "--", "invalid"])
        .output()
        .expect("Failed to execute CLI");

    // Should exit with error
    assert!(!output.status.success());
}

// Add more integration tests for specific commands
#[test]
fn test_cli_experiment_run() {
    // Placeholder: test running an experiment via CLI
    // Would create temp config, run command, check output files
}