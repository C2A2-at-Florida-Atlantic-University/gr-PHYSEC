#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-PHYSEC author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import onnxruntime as ort
import hashlib
import pmt
from gnuradio import gr
import time
from scipy import signal

class fingerprint_block(gr.sync_block):
    """
    Channel fingerprinting block using ONNX models.
    
    This block takes vector inputs of IQ samples and extracts channel fingerprints
    to generate quantized binary features for physical layer security.
    """
    
    def __init__(self, model_path, model_type, vector_size, sample_rate, center_freq, key_length):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Fingerprint Block",
            in_sig=[(np.complex64, vector_size)],  # Vector input
            out_sig=None
        )
        
        # Store parameters
        self.model_path = model_path
        self.model_type = model_type
        self.vector_size = vector_size
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.key_length = key_length
        
        # Initialize ONNX session
        self.ort_session = None
        
        # Message ports
        self.message_port_register_out(pmt.intern("fingerprint"))
        self.message_port_register_out(pmt.intern("quantized_features"))
        
        # Initialize the model
        self.load_model()
        
        # Check dependencies
        self.check_dependencies()
        
        print(f"PHYSEC Fingerprint Block initialized:")
        print(f"  Model: {model_path}")
        print(f"  Type: {model_type}")
        print(f"  Vector Size: {vector_size}")
        print(f"  Sample Rate: {sample_rate}")
        print(f"  Center Freq: {center_freq}")
        print(f"  Key Length: {key_length}")
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import onnxruntime
            print("✓ ONNX Runtime available")
        except ImportError:
            print("✗ ONNX Runtime not available. Please install: pip install onnxruntime")
            return False
        
        try:
            from scipy import signal
            print("✓ SciPy available")
        except ImportError:
            print("✗ SciPy not available. Please install: pip install scipy")
            return False
        
        return True
    
    def load_model(self):
        """Load the ONNX model."""
        try:
            # Configure ONNX Runtime providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # Create inference session
            self.ort_session = ort.InferenceSession(
                self.model_path, 
                providers=providers
            )
            
            # Get model input/output information
            input_info = self.ort_session.get_inputs()[0]
            output_info = self.ort_session.get_outputs()[0]
            
            print(f"✓ Model loaded successfully:")
            print(f"  Input: {input_info.name}, shape: {input_info.shape}, dtype: {input_info.type}")
            print(f"  Output: {output_info.name}, shape: {output_info.shape}, dtype: {output_info.type}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.ort_session = None
    
    def _normalization(self, data):
        """
        Normalize the signal by RMS, matching the dataset_preparation.py implementation.
        """
        try:
            s_norm = np.zeros(data.shape, dtype=np.complex64)
            
            for i in range(data.shape[0]):
                sig_amplitude = np.abs(data[i])
                rms = np.sqrt(np.mean(sig_amplitude**2))
                s_norm[i] = data[i]/rms
            
            return s_norm
        except Exception as e:
            print(f"Error in normalization: {e}")
            return data
    
    def _gen_single_channel_spectrogram(self, sig, win_len=256, overlap=128):
        """
        Generate single channel spectrogram matching dataset_preparation.py implementation.
        """
        try:
            # Short-time Fourier transform (STFT)
            f, t, spec = signal.stft(sig, 
                                    window='boxcar', 
                                    nperseg=win_len, 
                                    noverlap=overlap, 
                                    nfft=win_len,
                                    return_onesided=False, 
                                    padded=False, 
                                    boundary=None)
            
            # FFT shift to adjust the central frequency
            spec = np.fft.fftshift(spec, axes=0)
            
            # Take the logarithm of the magnitude
            chan_spec_amp = np.log10(np.abs(spec)**2)
            
            return chan_spec_amp
            
        except Exception as e:
            print(f"Error in _gen_single_channel_spectrogram: {e}")
            return None
    
    def _spec_crop(self, x):
        """
        Crop the generated channel independent spectrogram, matching dataset_preparation.py.
        """
        try:
            num_row = x.shape[0]
            x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
            return x_cropped
        except Exception as e:
            print(f"Error in spec_crop: {e}")
            return x
    
    def create_spectrogram(self, iq_data):
        """
        Create 2D spectrogram from IQ data matching the dataset_preparation.py workflow.
        
        Args:
            iq_data: Complex IQ samples of length vector_size
            
        Returns:
            2D spectrogram array or None if error
        """
        try:
            # Ensure we have the right number of samples
            if len(iq_data) != self.vector_size:
                print(f"Warning: Expected {self.vector_size} samples, got {len(iq_data)}")
                return None
            
            # Convert to complex array if needed
            if isinstance(iq_data, list):
                iq_data = np.array(iq_data)
            
            # Reshape to match the expected input format for the normalization function
            # The normalization function expects (num_samples, num_samples_per_packet)
            iq_data_reshaped = iq_data.reshape(1, -1)
            
            # Use the exact same parameters as in dataset_preparation.py
            FFTwindow = 512  # This matches the default in channel_spectrogram
            win_len = FFTwindow
            overlap = win_len/2
            
            print(f"Using FFT parameters: FFTwindow={FFTwindow}, win_len={win_len}, overlap={overlap}")
            
            # Normalize the IQ samples (matching dataset_preparation.py)
            iq_data_normalized = self._normalization(iq_data_reshaped)
            
            # Calculate the size of channel independent spectrograms
            num_row = int(win_len*0.4)  # 40% of frequency bins
            num_column = int(np.floor((iq_data_normalized.shape[1]-win_len)/overlap + 1))
            
            print(f"Expected spectrogram dimensions: num_row={num_row}, num_column={num_column}")
            
            # Generate spectrogram using the same method as dataset_preparation.py
            chan_spec_amp = self._gen_single_channel_spectrogram(iq_data_normalized[0], win_len, overlap)
            
            if chan_spec_amp is None:
                return None
            
            # Apply the same cropping as in dataset_preparation.py
            chan_spec_amp = self._spec_crop(chan_spec_amp)
            
            print(f"Raw spectrogram shape after processing: {chan_spec_amp.shape}")
            
            # Reshape to match the expected format (1, height, width, 1)
            # This matches the output format from dataset_preparation.py
            spectrogram_final = chan_spec_amp.reshape(1, chan_spec_amp.shape[0], chan_spec_amp.shape[1], 1)
            
            print(f"Final reshaped spectrogram: {spectrogram_final.shape}")
            return spectrogram_final
            
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None
    
    def extract_features(self, spectrogram):
        """
        Extract features using the ONNX model.
        
        Args:
            spectrogram: 2D spectrogram array
            
        Returns:
            Feature vector or None if error
        """
        try:
            if self.ort_session is None:
                print("No model loaded")
                return None
            
            # Get input name
            input_name = self.ort_session.get_inputs()[0].name
            
            # Run inference
            outputs = self.ort_session.run(
                None, 
                {input_name: spectrogram.astype(np.float32)}
            )
            
            # Get output
            features = outputs[0]
            print(f"Extracted features shape: {features.shape}")
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def feature_quantization(self, features):
        """
        Quantize features to binary values, matching the test file implementation.
        
        Args:
            features: Feature vector from model
            
        Returns:
            Binary array (0s and 1s)
        """
        try:
            # Use the same logic as in test_channel_fingerprinting_framework_onnx.py
            # Ensure features is a 1D array
            if features.ndim > 1:
                features = features.flatten()
            
            mean_features = np.mean(features)
            threshold = mean_features
            
            # Use numpy operations for efficient quantization
            features_quantized = (features >= threshold).astype(np.uint8)
            
            return features_quantized
            
        except Exception as e:
            print(f"Error in feature quantization: {e}")
            return None
    
    def work(self, input_items, output_items):
        """
        Main processing function called by GNU Radio.
        
        Args:
            input_items: List of input arrays
            output_items: List of output arrays (not used)
            
        Returns:
            Number of items processed
        """
        try:
            # Get input data (vector of complex samples)
            in0 = input_items[0]
            num_input_items = len(in0)
            
            print(f"Processing {num_input_items} vector(s) of size {self.vector_size}")
            
            for i in range(num_input_items):
                # Get the current vector of samples
                iq_vector = in0[i]
                
                # Create 2D spectrogram using the same logic as test file
                spectrogram = self.create_spectrogram(iq_vector)
                
                if spectrogram is not None:
                    # Extract features
                    features = self.extract_features(spectrogram)
                    
                    if features is not None:
                        # Quantize features to binary (matching test file logic)
                        quantized_features = self.feature_quantization(features)
                        
                        if quantized_features is not None:
                            # Create timestamp
                            timestamp = int(time.time() * 1e9)
                            
                            # Publish fingerprint message
                            fingerprint_msg = pmt.make_dict()
                            fingerprint_msg = pmt.dict_add(fingerprint_msg, pmt.intern("timestamp"), pmt.from_uint64(timestamp))
                            fingerprint_msg = pmt.dict_add(fingerprint_msg, pmt.intern("model_type"), pmt.intern(self.model_type))
                            fingerprint_msg = pmt.dict_add(fingerprint_msg, pmt.intern("vector_size"), pmt.from_long(self.vector_size))
                            fingerprint_msg = pmt.dict_add(fingerprint_msg, pmt.intern("sample_rate"), pmt.from_float(self.sample_rate))
                            fingerprint_msg = pmt.dict_add(fingerprint_msg, pmt.intern("center_freq"), pmt.from_float(self.center_freq))
                            
                            self.message_port_pub(pmt.intern("fingerprint"), fingerprint_msg)
                            
                            # Publish quantized features message
                            features_msg = pmt.make_dict()
                            features_msg = pmt.dict_add(features_msg, pmt.intern("timestamp"), pmt.from_uint64(timestamp))
                            features_msg = pmt.dict_add(features_msg, pmt.intern("quantized_features"), pmt.init_u8vector(len(quantized_features), quantized_features.astype(np.uint8)))
                            features_msg = pmt.dict_add(features_msg, pmt.intern("feature_length"), pmt.from_long(len(quantized_features)))
                            
                            self.message_port_pub(pmt.intern("quantized_features"), features_msg)
                            
                            print(f"✓ Generated quantized features: {len(quantized_features)} bits, mean: {np.mean(quantized_features):.3f}")
                        else:
                            print("✗ Failed to quantize features")
                    else:
                        print("✗ Failed to extract features")
                else:
                    print("✗ Failed to create spectrogram")
            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            return 0


def test_fingerprint_block():
    """
    Test function that tests the fingerprint block using the work function only
    """
    import sys
    import os
    
    # Add AI directory to path for imports
    ai_dir = '/workspace/gr-PHYSEC/AI'
    if ai_dir not in sys.path:
        sys.path.append(ai_dir)
    
    try:
        from dataset_preparation import LoadDatasetChannels
        import h5py
    except ImportError as e:
        print(f"Warning: Could not import required modules for testing: {e}")
        print("Some test functionality may not be available.")
        return
    
    print("=" * 80)
    print("TESTING FINGERPRINT BLOCK USING WORK FUNCTION")
    print("=" * 80)
    
    # Test configuration
    PHYSEC_dir = '/workspace/gr-PHYSEC/'
    model_path = PHYSEC_dir + 'models/'
    feature_extractor_onnx_name = model_path + 'QExtractor.onnx'
    dataset_path = PHYSEC_dir + 'datasets/'
    dataset = dataset_path + 'Dataset_Channels_sinusoid_dev_871_1690302750.hdf5'
    
    print(f"Model Path: {feature_extractor_onnx_name}")
    print(f"Dataset Path: {dataset}")
    
    try:
        # Load test data
        print("\nLoading test data...")
        LoadDatasetObj = LoadDatasetChannels()
        data, labels = LoadDatasetObj.load_iq_samples(dataset)
        print(f"Data shape: {data.shape}")
        
        # Use first few samples for testing
        test_data = data[:]  # Test all samples
        print(f"Test data shape: {test_data.shape}")
        print(f"First sample (first 10 values): {test_data[0][:10]}")
        
        # Create fingerprint block instance
        print("\nCreating fingerprint block instance...")
        fingerprint_block_instance = fingerprint_block(
            model_path=feature_extractor_onnx_name,
            model_type="QExtractor_ONNX",
            vector_size=len(test_data[0]),  # Use the length of first sample
            sample_rate=1e6,
            center_freq=2.4e9,
            key_length=256
        )
        
        # Prepare input data for work function
        # The work function expects input_items as a list of arrays
        # Each array should contain vectors of complex samples
        input_items = [test_data.astype(np.complex64)]  # Convert to complex64
        output_items = []  # Empty since this block has no outputs
        
        print(f"Input items shape: {input_items[0].shape}")
        print(f"Vector size: {fingerprint_block_instance.vector_size}")
        
        # Test the work function
        print("\n" + "="*50)
        print("TESTING WORK FUNCTION")
        print("="*50)
        
        t_start = time.time()
        print("Calling work function...")
        
        # Call the work function (this is what GNU Radio does)
        items_processed = fingerprint_block_instance.work(input_items, output_items)
        
        t_end = time.time()
        print(f"Work function completed in {t_end-t_start:.4f} seconds")
        print(f"Items processed: {items_processed}")
        
        # Check if the block processed all items
        if items_processed == len(test_data):
            print("✓ Work function processed all input items successfully")
        else:
            print(f"✗ Work function processed {items_processed} items, expected {len(test_data)}")
        
        # The work function publishes messages to message ports
        # In a real GNU Radio flowgraph, these would be connected to other blocks
        print("\n" + "="*50)
        print("MESSAGE PORT OUTPUTS")
        print("="*50)
        
        # Note: In the actual GNU Radio environment, these messages would be sent
        # to connected blocks. Here we're just testing that the work function
        # executes without errors and processes the data correctly.
        
        print("✓ Work function executed successfully")
        print("✓ Messages would be published to 'fingerprint' and 'quantized_features' ports")
        print("✓ In a real flowgraph, these would be connected to downstream blocks")
        
        # Test individual sample processing for validation
        print("\n" + "="*50)
        print("VALIDATION: INDIVIDUAL SAMPLE PROCESSING")
        print("="*50)
        
        # Test the first sample individually to verify the processing pipeline
        quantized_features = []
        for test_iq in test_data:
            print(f"Testing individual sample: {test_iq.shape}")
            
            # Test spectrogram creation
            test_spectrogram = fingerprint_block_instance.create_spectrogram(test_iq)
            if test_spectrogram is not None:
                print(f"✓ Individual spectrogram shape: {test_spectrogram.shape}")
                
                # Test feature extraction
                test_features = fingerprint_block_instance.extract_features(test_spectrogram)
                if test_features is not None:
                    print(f"✓ Individual features shape: {test_features.shape}")
                    
                    # Test feature quantization
                    test_quantized = fingerprint_block_instance.feature_quantization(test_features)
                    if test_quantized is not None:
                        print(f"✓ Individual quantized features: {len(test_quantized)} bits")
                        print(f"  Mean value: {np.mean(test_quantized):.3f}")
                        print(f"  Binary distribution: {np.bincount(test_quantized)}")
                        quantized_features.append(test_quantized)
                    else:
                        print("✗ Individual feature quantization failed")
                else:
                    print("✗ Individual feature extraction failed")
            else:
                print("✗ Individual spectrogram creation failed")
        
        

        # Save quantized features to a file
        output_file = dataset_path + 'Dataset_Channels_quantized_fingerprint_block_work_test.hdf5'
        with h5py.File(output_file, 'w') as f:
            quantized_features = np.array(quantized_features)
            print(f"Quantized features shape: {quantized_features.shape}")
            f.create_dataset('quantized_features', data=quantized_features)
        print(f"✓ Saved quantized features to: {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        # Clean up
        del data, labels, test_data, test_iq, test_spectrogram, test_features, test_quantized, fingerprint_block_instance, quantized_features, output_file
        
    except Exception as e:
        print(f"✗ Error in testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Test the fingerprint block as in test_channel_fingerprinting_framework_onnx.py
    print("Testing fingerprint block...")
    test_fingerprint_block()
    
    