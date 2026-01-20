"""
End-to-end test for liquid manifold generation pipeline.
"""

import pytest
import tempfile
import os
import json
import subprocess
import sys


def test_liquid_manifold_generation_execution():
    """Test complete execution of liquid manifold generation pipeline."""
    # Create minimal config for liquid manifold generation
    config_content = """
[simulation_parameters]
exc_only = true
on_phase = 500  # Small for testing
off_phase = 1000
settling_period = 500
tolerance = 2
peaks_on = true
trials = 2  # Small number
filename = "test_manifold_output.json"

[variables]
input_table = [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]]  # 3x3 input
spike_train_connectivity = [1.0]
spike_train_to_exc = [3]
glutamate_clearance = [0.001]
gabaa_clearance = [0.001]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "test_manifold_config.toml")
        output_file = os.path.join(temp_dir, "test_manifold_output.json")

        with open(config_file, 'w') as f:
            f.write(config_content)

        # Run the pipeline
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "liquid_custom_manifold_generation.py")
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
        assert 'return_to_baseline' in entry, "Entry should contain return_to_baseline"
        assert 'voltages' in entry, "Entry should contain voltages"
        assert isinstance(entry['voltages'], list), "voltages should be a list"
        assert len(entry['voltages']) > 100, "Should have voltage data points"


def test_liquid_manifold_generation_error_handling():
    """Test error handling in liquid manifold generation."""
    # Config with missing required field
    invalid_config = """
[simulation_parameters]
# Missing filename

[variables]
input_table = [[[0.1]]]
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "invalid_manifold_config.toml")

        with open(config_file, 'w') as f:
            f.write(invalid_config)

        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "liquid_custom_manifold_generation.py")
        result = subprocess.run([sys.executable, script_path, config_file], cwd=temp_dir, capture_output=True, text=True)

        # Should fail with error
        assert result.returncode != 0, "Pipeline should fail with invalid config"