"""
End-to-end test for GPU bayesian inference pipeline.
"""

import pytest
import tempfile
import os
import json
import subprocess
import sys


def test_bayesian_inference_pipeline_execution():
    """Test complete execution of GPU bayesian inference pipeline."""
    # Create minimal config with small iterations for fast testing
    config_content = """
[simulation_parameters]
peaks_on = false
bayesian_is_not_main = true
use_correlation_as_accuracy = true
measure_snr = false
iterations1 = 500  # Small for testing
iterations2 = 0
a = -1
b = 0
weights_scalar = 1
inh_weights_scalar = 1
skew = 0.1
c_m = 25
trials = 2  # Small number
gpu_batch = 2
filename = "test_bayesian_output.json"

[variables]
spike_train_to_exc = [4]
bayesian_to_exc = [0.5]
s_d2 = [0]
s_d1 = [0]
prob_of_exc_to_inh = [1]
distortion = [0.2]
bayesian_distortion = [0]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "test_bayesian_config.toml")
        output_file = os.path.join(temp_dir, "test_bayesian_output.json")

        with open(config_file, 'w') as f:
            f.write(config_content)

        # Run the pipeline - need to set PYTHONPATH for interface_gpu
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "bayesian_inference_pipeline.py")
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), "..", "..") + ':' + env.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, script_path, config_file], cwd=temp_dir, env=env, capture_output=True, text=True)

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


def test_bayesian_inference_pipeline_error_handling():
    """Test error handling in GPU bayesian inference pipeline."""
    # Config with missing required field
    invalid_config = """
[simulation_parameters]
# Missing filename
iterations1 = 500

[variables]
spike_train_to_exc = [4]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "invalid_bayesian_config.toml")

        with open(config_file, 'w') as f:
            f.write(invalid_config)

        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "bayesian_inference_pipeline.py")
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), "..", "..") + ':' + env.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, script_path, config_file], cwd=temp_dir, env=env, capture_output=True, text=True)

        # Should fail with error
        assert result.returncode != 0, "Pipeline should fail with invalid config"