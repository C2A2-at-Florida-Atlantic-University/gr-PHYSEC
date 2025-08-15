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

# Create fingerprint block
fingerprint = PHYSEC.fingerprint_block(
    model_path="/path/to/QExtractor.h5",
    model_type="quadruplet",
    spectrogram_size=512,
    sample_rate=1e6,
    center_freq=2.4e9,
    key_length=128
)
```

### Example Flowgraph

See `examples/fingerprint_example.grc` for a complete example flowgraph.

## Block Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | - | Path to QExtractor.h5 model file |
| `model_type` | string | "quadruplet" | Model type: "quadruplet" or "triplet" |
| `spectrogram_size` | int | 512 | FFT size for spectrogram creation |
| `sample_rate` | float | 1e6 | SDR sample rate in Hz |
| `center_freq` | float | 2.4e9 | SDR center frequency in Hz |
| `key_length` | int | 128 | Desired cryptographic key length |

## Input/Output

### Input
- **Stream Input**: Complex IQ samples from SDR

### Output (Message Ports)
- **`fingerprint_out`**: Generated cryptographic fingerprint
- **`features_out`**: Quantized features for debugging
- **`spectrogram_out`**: Spectrogram data for debugging

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

1. **IQ Sample Collection**: Buffer incoming IQ samples
2. **Spectrogram Creation**: Convert IQ samples to spectrogram using FFT
3. **Feature Extraction**: Use trained model to extract channel features
4. **Feature Quantization**: Convert features to binary values
5. **Privacy Amplification**: Apply SHA3-512 hashing for key generation
6. **Output**: Send results via message ports

## Model Requirements

Your QExtractor.h5 model should:
- Accept input shape: `(batch, height, width, channels)` or `(batch, 1, spectrogram_size, 1)`
- Output feature vectors that can be quantized
- Be compatible with TensorFlow 2.x

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

1. Modify `fingerprint_block.py`
2. Update the GRC block definition in `grc/`
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
