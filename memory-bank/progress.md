# Progress

This file tracks the project's progress using a task list format.
YYYY-MM-DD HH:MM:SS - Log of updates made.
*

[2026-01-20 11:01:30] - Added integration tests for release: Python CPU interface tests in interface/tests/, CLI integration tests in ndfh-cli, end-to-end experiment tests, and error handling tests.

## Completed Tasks

[2026-01-20 13:01:00] - Added API stability documentation to docs/api/rust-api.md and docs/api/python-api.md with stability indicators, version compatibility guarantees, deprecation notices, and migration guides.

[2026-01-20 16:40:00] - Created end-to-end tests for complete experiment pipelines. Added automated tests in interface/tests/e2e/ and interface_gpu/tests/e2e/ for bayesian_inference_pipeline, liquid_manifold_generation, and schizophrenia_simulation_pipeline. Tests validate complete pipeline execution, output validation, and error handling for both CPU and GPU variants.

*

## Current Tasks

[2026-01-20 11:08:48] - Validated and updated GPU support documentation for release accuracy. Updated docs to reflect OpenCL-only support, corrected API examples, documented limitations (no CUDA, no chemical synapses on GPU, no plasticity on GPU, etc.).

## Next Steps

*