# gr-PHYSEC: GNU Radio Module for Physical Layer Security

This GNU Radio module provides channel fingerprinting capabilities for physical layer security applications using deep learning models.

## Features

- **Channel Fingerprinting**: Extract unique channel characteristics from IQ samples
- **Deep Learning Integration**: Uses trained models (QExtractor.h5) for feature extraction
- **Feature Quantization**: Convert extracted features to binary values
- **Privacy Amplification**: Apply cryptographic hashing (SHA3-512) for key generation
- **Real-time Processing**: Process IQ samples in real-time from SDR devices
- **Multiple Model Support**: Support for both TripletNet and QuadrupletNet architectures

## Requirements

- GNU Radio 3.10+
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Hashlib (built-in)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>/gr-PHYSEC.git
cd gr-PHYSEC
```

### 2. Create build directory
```bash
mkdir build
cd build
```

### 3. Configure and build
```bash
cmake ..
make -j$(nproc)
```

### 4. Install
```bash
sudo make install
sudo ldconfig
```

### 5. Verify installation
```bash
gnuradio-companion
```
You should see a new category `[PHYSEC]` with the `PHYSEC Fingerprint Block`.

## Usage

### GNU Radio Companion (GRC)

1. Open GNU Radio Companion
2. Look for the `[PHYSEC]` category
3. Drag and drop the `PHYSEC Fingerprint Block`
4. Configure the parameters:
   - **Model Path**: Path to your QExtractor.h5 file
   - **Model Type**: Choose between "quadruplet" or "triplet"
   - **Spectrogram Size**: FFT size (default: 512)
   - **Sample Rate**: SDR sample rate in Hz
   - **Center Frequency**: SDR center frequency in Hz
   - **Key Length**: Desired key length in bits

### Python Script

```python
from gnuradio import gr
from gnuradio import PHYSEC

# Create decoupled PHYSEC blocks
spectrogram = PHYSEC.spectrogram_block(
    vector_size=512,
    sample_rate=1e6,
    center_freq=2.4e9
)

feature_extractor = PHYSEC.feature_extraction_block(
    model_path="/path/to/QExtractor.onnx"
)

feature_quantizer = PHYSEC.feature_quantization_block(
    threshold_type="mean"
)

parity_generator = PHYSEC.parity_generation_block(
    key_length=512
)

reconciler = PHYSEC.reconciliation_block(
    key_length=512
)

privacy_amplifier = PHYSEC.privacy_amplification_block()
```

### Example Flowgraph

See `examples/decoupled_physic_example.grc` for a complete example flowgraph using the decoupled PHYSEC blocks.

## Block Parameters

### Spectrogram Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_size` | int | 512 | FFT size for spectrogram creation |
| `sample_rate` | float | 1e6 | SDR sample rate in Hz |
| `center_freq` | float | 2.4e9 | SDR center frequency in Hz |

### Feature Extraction Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | - | Path to ONNX model file |

### Feature Quantization Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_type` | string | "mean" | Threshold method: "mean", "median", or "zero" |

### Parity Generation Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_length` | int | 512 | Length of binary key for parity generation |

### Reconciliation Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_length` | int | 512 | Length of binary key for reconciliation |

### Privacy Amplification Block
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| No parameters required | - | - | Applies SHA3-512 hashing |

## Input/Output

### Spectrogram Block
- **Input**: Complex IQ samples from SDR
- **Output**: Spectrogram data (204×31 float values)

### Feature Extraction Block
- **Input**: Spectrogram data (204×31 float values)
- **Output**: Feature vector (512 float values)

### Feature Quantization Block
- **Input**: Feature vector (512 float values)
- **Output**: Binary features (512 uint8 values)

### Parity Generation Block
- **Input**: Binary features (512 uint8 values)
- **Output**: Parity bits (string)

### Reconciliation Block
- **Input**: Binary features + Parity bits
- **Output**: Reconciled binary key (string)

### Privacy Amplification Block
- **Input**: Binary features (512 uint8 values)
- **Output**: Final cryptographic key (128 bytes)

## Message Format

The fingerprint output message contains:
```json
{
    "fingerprint": "generated_hash_string",
    "block_count": 123,
    "sample_rate": 1000000.0,
    "center_freq": 2400000000.0,
    "spectrogram_size": 512
}
```

## Processing Pipeline

1. **IQ Sample Collection**: Buffer incoming IQ samples using Stream to Vector
2. **Spectrogram Creation**: Convert IQ samples to spectrogram using STFT, RMS normalization, and frequency cropping
3. **Feature Extraction**: Use ONNX model to extract channel features
4. **Feature Quantization**: Convert features to binary values using configurable thresholds
5. **Parity Generation**: Generate Reed-Solomon parity bits for error correction
6. **Key Reconciliation**: Correct errors using parity bits
7. **Privacy Amplification**: Apply SHA3-512 hashing for final key generation
8. **Output**: Stream-based communication between blocks

## Model Requirements

Your ONNX model should:
- Accept input shape: `(batch, 204, 31)` for spectrogram data
- Output feature vectors of length 512
- Be compatible with ONNX Runtime
- Support float32 input/output

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the path to QExtractor.h5 is correct
2. **TensorFlow errors**: Verify TensorFlow installation and compatibility
3. **Memory issues**: Reduce spectrogram_size for lower memory usage
4. **Performance**: Use GPU acceleration if available

### Debug Output

The block provides extensive debug information:
- Console output for processing status
- Message ports for intermediate results
- Error handling with informative messages

## Development

### Adding New Features

1. Modify the appropriate specialized block (e.g., `spectrogram_block.py`)
2. Update the corresponding GRC block definition in `grc/`
3. Rebuild the module
4. Test with examples

### Testing

Run the example script:
```bash
cd examples
python3 fingerprint_example.py
```

## License

This project is licensed under the GPL-3.0-or-later License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review example files
3. Open an issue on the repository
4. Contact the development team

## Acknowledgments

- Based on the PHYSEC channel fingerprinting research
- Uses TensorFlow for deep learning inference
- Integrates with GNU Radio for SDR processing
