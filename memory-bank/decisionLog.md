# Decision Log

This file records architectural and implementation decisions using a list format.
YYYY-MM-DD HH:MM:SS - Log of updates made.

*

## Decision

[2026-01-20 12:58:00] - Simplified installation process to prioritize one primary path for scientific users

## Rationale

Current installation has multiple complex paths (nightly Rust, maturin, GPU dependencies) which confuse users. Scientific users primarily need Python interface, so prioritize pip/maturin path as most accessible. This reduces barrier to entry while keeping advanced options available.

## Implementation Details

- Primary path: Python via pip install maturin + maturin develop --release
- Prerequisites: Python 3.8+, pip, git (Rust handled automatically)
- Optional: Rust dev setup, GPU support
- Updated INSTALL.md and README.md to reflect streamlined approach
- Added troubleshooting for common issues

*

[2026-01-20 13:01:00] - Added API stability documentation to clearly communicate stable vs experimental APIs

## Rationale

As SHNN matures, users need clear guidance on API stability to make informed decisions about production use. Without stability indicators, users risk unexpected breaking changes in experimental APIs.

## Implementation Details

- Classified APIs as Stable (core neuron models, basic network operations), Experimental (advanced features, hardware acceleration), or Unstable (none currently)
- Added stability badges using Shields.io in docs/api/rust-api.md and docs/api/python-api.md
- Added version compatibility guarantees section explaining semver policy
- Added deprecation notices section (currently empty)
- Added migration guides section with examples for future breaking changes
- Focused on core APIs in both Rust and Python interfaces

*

*