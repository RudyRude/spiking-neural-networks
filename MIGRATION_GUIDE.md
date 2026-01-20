# Migration Guide

This guide helps users transition from the previous separate repositories (`hSNN/`, `ndf-h/`) to the unified `spiking-neural-networks/` framework.

## Overview

The spiking neural network ecosystem has been consolidated into a single, unified framework that combines:

- Core SNN functionality from `hSNN/`
- Data format support from `ndf-h/`
- Enhanced Python bindings
- Improved CLI tools
- Modular architecture

## Key Changes

### Unified Repository Structure

**Before:**
```
hSNN/           # Core SNN implementation
ndf-h/          # Data formats
(hSNN copy/)    # Alternative implementation
```

**After:**
```
spiking-neural-networks/
├── backend/    # Rust implementation
│   ├── crates/
│   │   ├── shnn-core/     # Neuron models
│   │   ├── shnn-cli/      # CLI tools
│   │   ├── shnn-python/   # Python bindings
│   │   ├── ndfh-*/        # Data format crates
│   │   └── ...
└── docs/       # Documentation
```

### API Changes

#### Python API

**Old API:**
```python
# From separate hSNN package
import hsnn
network = hsnn.Network()
```

**New API:**
```python
import shnn

# Create network
network = shnn.Network()

# Add neurons with new models
network.add_neuron(shnn.LIFNeuron(tau_m=20.0, v_thresh=-50.0))
network.add_neuron(shnn.AdExNeuron())

# Connect with hypergraph support
network.connect(0, 1, weight=0.5, plasticity=shnn.STDP())

# Run simulation
spikes = network.run(timesteps=1000)

# Export in multiple formats
network.export_spikes("output.graphml")
network.export_spikes("output.lpg.json")
```

#### CLI Changes

**Old CLI:**
```bash
# Limited format support
hsnn simulate network.json
```

**New CLI:**
```bash
# Enhanced CLI with NIR compilation
shnn nir compile network.nir -o compiled.json

# Multiple export formats
shnn export spikes.vmsk --format graphml
shnn export spikes.vmsk --format rdf-nquads
```

## Migration Steps

### 1. Update Dependencies

**Python:**
```bash
# Remove old packages
pip uninstall hsnn ndfh

# Install unified package
pip install spiking-neural-networks
```

**Rust:**
```toml
# Cargo.toml
[dependencies]
spiking-neural-networks = "0.1.0"  # New unified crate
```

### 2. Update Code

#### Neuron Model Migration

**Old:**
```python
# Basic neuron
neuron = hsnn.LIF()
```

**New:**
```python
# Specify parameters explicitly
neuron = shnn.LIFNeuron(tau_m=20.0, v_reset=-70.0, v_thresh=-50.0)
```

#### Connectivity Migration

**Old:**
```python
# Simple connections
network.connect(0, 1, weight=0.5)
```

**New:**
```python
# Enhanced connectivity options
network.connect(0, 1, weight=0.5, delay=1, plasticity=shnn.STDP(eta=0.01))

# Hypergraph support
network.add_hyperedge([0, 1, 2], weight=0.3)
```

### 3. Data Format Migration

**Old:**
```python
# Limited JSON export
network.save("network.json")
```

**New:**
```python
# Multiple formats supported
network.export_network("network.graphml")
network.export_network("network.lpg.json")
network.export_spikes("spikes.vmsk")
```

### 4. Configuration Migration

**Old configuration files** from `hSNN/` and `ndf-h/` should be converted to use the new unified format.

**Example conversion:**
```json
// Old format
{
  "neurons": [{"type": "lif", "params": {...}}],
  "connections": [{"from": 0, "to": 1, "weight": 0.5}]
}

// New format (NIR compatible)
{
  "nodes": [
    {"id": "n0", "type": "lif", "parameters": {...}},
    {"id": "n1", "type": "lif", "parameters": {...}}
  ],
  "edges": [
    {"source": "n0", "target": "n1", "weight": 0.5}
  ]
}
```

## Breaking Changes

### Removed Features
- Deprecated neuron models without explicit parameters
- Old binary format (use VEVT/VMSK instead)
- Separate repository dependencies

### New Features
- NDF-H data format support (GraphML, LPG-JSON, RDF-NQuads)
- NIR compilation for optimized execution
- Hypergraph connectivity
- Plasticity models (STDP, etc.)
- Microcontroller support (`shnn-micro`)

## Troubleshooting

### Import Errors
If you encounter import errors after migration:

```python
# Update imports
import shnn as snn  # Instead of hsnn
```

### Performance Issues
The new framework includes performance optimizations:

- Use `shnn-micro` for embedded systems
- Enable NIR compilation for faster execution
- Use sparse matrices for large networks

### Data Compatibility
Old data files may need conversion:

```bash
# Convert old format to new
shnn convert old_network.json --to nir
```

## Support

For migration assistance:
- Check the updated documentation in `docs/`
- Review examples in `backend/examples/`
- File issues on the unified repository

## Timeline

- **Phase 1**: Core migration (neuron models, basic connectivity)
- **Phase 2**: Advanced features (plasticity, hypergraphs)
- **Phase 3**: Data format integration
- **Phase 4**: Performance optimizations

The unified framework maintains backward compatibility where possible while providing a modern, extensible API for spiking neural network research and applications.