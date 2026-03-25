# PHYSEC Flowgraphs

This directory contains GNU Radio Companion (GRC) flowgraphs for the PHYSEC key generation protocol.

## Working GRC Files

### ✅ **Alice and Bob Node Flowgraphs (YAML Format)**

- **`alice_node_yaml.grc`** - Complete Alice node flowgraph in YAML format for GNU Radio 3.10
- **`bob_node_yaml.grc`** - Complete Bob node flowgraph in YAML format for GNU Radio 3.10

These files use the **correct YAML format** that GNU Radio 3.10 expects and include all the necessary blocks for SDR communication and signal processing.

#### **Block Structure:**
- **Variables**: `sample_rate`, `sdr_uri`, `vector_size`
- **IIO Blocks**: `iio_fmcomms2_source_fc32`, `iio_fmcomms2_sink_fc32`
- **Signal Generation**: `analog_sig_source_c`
- **Processing**: `blocks_head`, `blocks_stream_to_vector`
- **Data Collection**: `blocks_vector_sink_c` (2 instances)

#### **Connections:**
1. PlutoSDR Source → Head → Stream to Vector → Vector Sink (Vector)
2. PlutoSDR Source → Head → Vector Sink (IQ)
3. Signal Source → PlutoSDR Sink

### ✅ **Individual Component Flowgraphs**

- **`physec_processor.grc`** - PHYSEC signal processing pipeline
- **`sinusoidal_probe.grc`** - Sinusoidal probe generation
- **`iq_receiver.grc`** - IQ sample collection
- **`parity_generator.grc`** - Parity bit generation
- **`reconciliator.grc`** - Key reconciliation
- **`privacy_amplifier.grc`** - Privacy amplification

## Python Implementations

- **`alice_node.py`** - Full Python implementation of Alice node with PHYSEC
- **`bob_node.py`** - Full Python implementation of Bob node with PHYSEC
- **`physec_processor.py`** - PHYSEC processing flowgraph
- **`sinusoidal_probe.py`** - Probe generation flowgraph
- **`iq_receiver.py`** - IQ collection flowgraph
- **`reconciliator.py`** - Reconciliation flowgraph
- **`privacy_amplifier.py`** - Privacy amplification flowgraph
- **`parity_generator.py`** - Parity generation flowgraph

## Usage

### **In GNU Radio Companion:**
1. **Open GRC**: Launch GNU Radio Companion
2. **Load Flowgraph**: File → Open → Select `alice_node_yaml.grc` or `bob_node_yaml.grc`
3. **Modify Parameters**: Adjust sample rate, SDR URI, vector size as needed
4. **Generate**: Generate → Generate Python
5. **Run**: Execute the generated Python file

### **Direct Python Usage:**
```python
from flowgraphs import create_alice_node_flowgraph, create_bob_node_flowgraph

# Create and run Alice node
alice = create_alice_node_flowgraph()
alice.start()
alice.wait()

# Create and run Bob node
bob = create_bob_node_flowgraph()
bob.start()
bob.wait()
```

## Key Features

- **SDR Communication**: PlutoSDR source/sink for IQ collection and probe transmission
- **Signal Processing**: Stream-to-vector conversion and sample collection
- **Data Collection**: Vector sinks for storing IQ samples and processed data
- **Probe Generation**: Sine wave generation for probe signals
- **GNU Radio 3.10 Compatible**: Uses correct block IDs and YAML format

## Block IDs Used

The working GRC files use the exact block IDs available in GNU Radio 3.10:

- **IIO**: `iio_fmcomms2_source_fc32`, `iio_fmcomms2_sink_fc32`
- **Blocks**: `blocks_head`, `blocks_stream_to_vector`, `blocks_vector_sink_c`
- **Analog**: `analog_sig_source_c`

## Troubleshooting

### **"Missing Block" Errors:**
- Ensure you're using GNU Radio 3.10
- Use the YAML format files (`*_yaml.grc`)
- Check that all block IDs match your installation

### **YAML Parsing Errors:**
- Use the `*_yaml.grc` files (not the old XML format)
- Ensure proper YAML syntax and indentation

### **Block Not Found:**
- Verify block IDs match your GNU Radio installation
- Check that required modules are installed (IIO, blocks, analog)

## File Formats

- **`.grc`** - GNU Radio Companion flowgraph files
- **`.py`** - Generated Python implementations
- **YAML Format** - Modern GNU Radio 3.10 format (recommended)
- **XML Format** - Legacy format (deprecated in GR 3.10)

## Notes

- The YAML format files are the **recommended** format for GNU Radio 3.10
- All working flowgraphs use the **exact block IDs** available in your installation
- The Python implementations provide full PHYSEC protocol functionality
- These flowgraphs are designed for **PlutoSDR** hardware compatibility
