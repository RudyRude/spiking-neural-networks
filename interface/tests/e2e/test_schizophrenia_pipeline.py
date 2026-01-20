"""
End-to-end test for schizophrenia simulation pipeline.
"""

import pytest
import tempfile
import os
import json
import subprocess
import sys


def test_schizophrenia_pipeline_execution():
    """Test complete execution of schizophrenia simulation pipeline."""
    # Create minimal config with small iterations for fast testing
    config_content = """
[simulation_parameters]
peaks_on = true
second_cue = false
use_correlation_as_accuracy = false
measure_snr = false
weights_scalar = 1
inh_weights_scalar = 1
skew = 0.1
c_m = 25
a = -1
b = 0
first_window = 100
iterations1 = 500  # Small for testing
iterations2 = 0
trials = 2  # Small number of trials
filename = "test_output.json"

[variables]
spike_train_to_exc = [4.5]
prob_of_exc_to_inh = [1]
glutamate_clearance = [0.005]
gabaa_clearance = [0.005]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "test_config.toml")
        output_file = os.path.join(temp_dir, "test_output.json")

        with open(config_file, 'w') as f:
            f.write(config_content)

        # Run the pipeline
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "schizophrenia_simulation_pipeline.py")
        result = subprocess.run([sys.executable, script_path, config_file], cwd=temp_dir, capture_output=True, text=True)

        # Check that it ran successfully
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Check output file was created
        assert os.path.exists(output_file), "Output JSON file was not created"

        # Load and validate output structure
        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, dict), "Output should be a dictionary"
        assert len(data) > 0, "Output should contain results"

        # Check structure of first entry
        first_key = list(data.keys())[0]
        entry = data[first_key]
        assert 'first_acc' in entry, "Entry should contain first_acc"
        assert isinstance(entry['first_acc'], (int, float)), "first_acc should be numeric"
        assert 'peaks' in entry, "Entry should contain peaks when peaks_on=true"


def test_schizophrenia_pipeline_error_handling():
    """Test error handling in schizophrenia pipeline."""
    # Config with missing required field
    invalid_config = """
[simulation_parameters]
# Missing filename
iterations1 = 500

[variables]
spike_train_to_exc = [4.5]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "invalid_config.toml")

        with open(config_file, 'w') as f:
            f.write(invalid_config)

        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "schizophrenia_simulation_pipeline.py")
        result = subprocess.run([sys.executable, script_path, config_file], cwd=temp_dir, capture_output=True, text=True)

        # Should fail with error
        assert result.returncode != 0, "Pipeline should fail with invalid config"