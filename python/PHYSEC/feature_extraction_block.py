#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-PHYSEC author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import onnxruntime as ort
from gnuradio import gr
import time

class feature_extraction_block(gr.sync_block):
    """
    Feature extraction block using ONNX models.
    
    This block takes spectrogram inputs and extracts features using a pre-trained
    neural network model, matching the test_channel_fingerprinting_framework_onnx.py logic
    """
    
    def __init__(self, model_path):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Feature Extraction Block",
            in_sig=[(np.float32, (204,31))],  # spectrogram input (204,31)
            out_sig=[(np.float32, 512)]  # Feature vector (512 features)
        )
        
        # Store parameters
        self.model_path = model_path
        
        # Initialize ONNX session
        self.ort_session = None
        
        # Initialize the model
        self.load_model()
        
        # Check dependencies
        self.check_dependencies()
        
        print(f"PHYSEC Feature Extraction Block initialized:")
        print(f"  Model: {model_path}")
        print(f"  Input Size: 204,31 (spectrogram)")
        print(f"  Output Size: 512 (features)")
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import onnxruntime
            print("✓ ONNX Runtime available")
        except ImportError:
            print("✗ ONNX Runtime not available. Please install: pip install onnxruntime")
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
    
    def extract_features(self, spectrogram):
        """
        Extract features using the ONNX model.
        
        Args:
            spectrogram: Flattened spectrogram array
            
        Returns:
            Feature vector or None if error
        """
        try:
            if self.ort_session is None:
                print("No model loaded")
                return None
            
            # Get input name
            input_name = self.ort_session.get_inputs()[0].name
            
            spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)
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
    
    def work(self, input_items, output_items):
        """
        Main processing function called by GNU Radio.
        
        Args:
            input_items: List of input arrays
            output_items: List of output arrays
            
        Returns:
            Number of items processed
        """
        try:
            # Get input and output data
            in0 = input_items[0]
            out0 = output_items[0]
            num_input_items = len(in0)
            
            print(f"Processing {num_input_items} spectrogram(s)")
            
            for i in range(num_input_items):
                # Get the current flattened spectrogram
                spectrogram_flat = in0[i]
                
                # Extract features
                features = self.extract_features(spectrogram_flat)
                    
                out0[i] = features.astype(np.float32)
                print(f"✓ Extracted features: {features.shape}")
            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            return 0


if __name__ == "__main__":
    # Test the feature extraction block
    print("Testing feature extraction block...")
    
    # Create test data (mock flattened spectrogram)
    test_spectrogram = np.random.randn(204, 31).astype(np.float32)
    
    # Create block instance
    model_path = "/workspace/gr-PHYSEC/models/QExtractor.onnx"
    block = feature_extraction_block(model_path)
    
    # Test feature extraction
    features = block.extract_features(test_spectrogram)
    if features is not None:
        print(f"✓ Test successful! Features shape: {features.shape}")
    else:
        print("✗ Test failed!")
