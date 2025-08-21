# PHYSEC GNU Radio Flowgraphs

This directory contains modular GNU Radio flowgraph implementations for the PHYSEC quantum key generation protocol. Each flowgraph is self-contained and can be used independently or as part of the larger control layer system.

## 📁 Flowgraph Files

### 🔧 Core Processing Flowgraphs

| **File** | **Description** | **Purpose** |
|----------|-----------------|-------------|
| `physec_processor.py` | PHYSEC signal processing pipeline | IQ → Spectrogram → Features → Quantized Bits |
| `sinusoidal_probe.py` | Sinusoidal signal transmission | Generate and transmit 1 kHz probe signal |
| `iq_receiver.py` | IQ sample collection | Receive and collect IQ samples from SDR |

### 🔐 Cryptographic Flowgraphs

| **File** | **Description** | **Purpose** |
|----------|-----------------|-------------|
| `parity_generator.py` | Reed-Solomon parity generation | Generate error correction parity bits |
| `reconciliator.py` | Reed-Solomon reconciliation | Correct errors using parity bits |
| `privacy_amplifier.py` | Privacy amplification | Generate final cryptographic key |

### 📋 Support Files

| **File** | **Description** | **Purpose** |
|----------|-----------------|-------------|
| `__init__.py` | Module initialization | Factory functions and imports |
| `*.grc` | GNU Radio Companion files | Visual flowgraph representations |

## 🚀 Usage

### Direct Import (Recommended)

```python
from flowgraphs import (
    create_physec_processor,
    create_sinusoidal_probe,
    create_iq_receiver,
    create_parity_generator,
    create_reconciliator,
    create_privacy_amplifier
)

# Create and use flowgraphs
processor = create_physec_processor(iq_samples)
probe = create_sinusoidal_probe(sample_rate=1e6, frequency=1000)
```

### Individual Module Import

```python
from flowgraphs.physec_processor import PhysecProcessor
from flowgraphs.sinusoidal_probe import SinusoidalProbe

# Use classes directly
processor = PhysecProcessor(samples, fft_window=512, vector_size=8192)
probe = SinusoidalProbe(sample_rate=1e6, frequency=1000, amplitude=0.5)
```

### Standalone Testing

Each flowgraph can be tested independently:

```bash
cd flowgraphs/
python3 physec_processor.py      # Test PHYSEC processing
python3 sinusoidal_probe.py      # Test signal transmission  
python3 iq_receiver.py           # Test sample collection
python3 parity_generator.py      # Test parity generation
python3 reconciliator.py         # Test reconciliation
python3 privacy_amplifier.py     # Test privacy amplification
```

## 📊 Flowgraph Details

### PhysecProcessor
**Input**: Complex IQ samples (8192 samples)  
**Output**: Quantized bits (512 bits) + Spectrogram data (204×31)  
**Blocks**: Spectrogram → Feature Extraction → Quantization  

### SinusoidalProbe  
**Input**: Configuration parameters  
**Output**: RF signal via PlutoSDR (or file)  
**Blocks**: Signal Source → Noise → PlutoSDR Sink  

### IQReceiver
**Input**: RF signal via PlutoSDR  
**Output**: Complex IQ samples (8192 samples)  
**Blocks**: PlutoSDR Source → Head → Vector Sink  

### ParityGenerator
**Input**: Binary key (512 bytes)  
**Output**: Parity bits (127 bytes)  
**Blocks**: Vector Source → Reed-Solomon Encoder → File Sink  

### Reconciliator  
**Input**: Binary key + Parity bits  
**Output**: Reconciled key (128 bytes) + Success flag  
**Blocks**: Vector Sources → Reed-Solomon Decoder → File Sinks  

### PrivacyAmplifier
**Input**: Reconciled key (128 bytes)  
**Output**: Final cryptographic key (128 bytes)  
**Blocks**: Vector Source → SHA3-512 Hash → File Sink  

## 🔧 Requirements

- **GNU Radio 3.10+**
- **gr-PHYSEC library** (custom PHYSEC blocks)
- **PlutoSDR** (optional, falls back to test signals/file I/O)
- **Python 3.x** with numpy

## 🎯 Integration

These flowgraphs are automatically imported and used by `control_layer.py`:

```python
# Automatic selection in control_layer.py
if FLOWGRAPHS_AVAILABLE:
    # Use modular flowgraphs (this directory)
    processor = create_physec_processor(samples)
else:
    # Fallback to inline implementations
    processor = InlinePhysecProcessor(samples)
```

## 📈 Benefits of Modular Design

✅ **Reusability**: Each flowgraph can be used independently  
✅ **Testability**: Individual components can be tested in isolation  
✅ **Maintainability**: Easier to debug and modify specific functions  
✅ **Visualization**: GRC files provide visual flowgraph representation  
✅ **Flexibility**: Easy to swap implementations or add new features  

## 🔍 GRC Visualization

GNU Radio Companion (`.grc`) files are provided for visual flowgraph editing:

```bash
gnuradio-companion sinusoidal_probe.grc
```

This opens the flowgraph in the GNU Radio visual editor for inspection and modification.

## 🧪 Testing

The entire modular system can be tested with:

```bash
cd ../
python3 test_modular_system.py  # Test integration
python3 demo_control_layer.py --run --runs 1  # Full protocol test
```
