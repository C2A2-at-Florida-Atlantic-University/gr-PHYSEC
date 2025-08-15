#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-PHYSEC author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr
import time

class feature_quantization_block(gr.sync_block):
    """
    Feature quantization block.
    
    This block takes continuous feature vectors and converts them to binary values
    using the same logic as test_channel_fingerprinting_framework_onnx.py
    """
    
    def __init__(self, quantization_method="mean_threshold"):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Feature Quantization Block",
            in_sig=[(np.float32, 512)],  # Feature vector input
            out_sig=[(np.uint8, 512)]  # Binary array output
        )
        
        # Store parameters
        self.quantization_method = quantization_method
        
        print(f"PHYSEC Feature Quantization Block initialized:")
        print(f"  Quantization Method: {quantization_method}")
        print(f"  Input Size: 512 (features)")
        print(f"  Output Size: 512 (binary)")
    
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
            
            if self.quantization_method == "mean_threshold":
                # Use mean as threshold (default method from test file)
                mean_features = np.mean(features)
                threshold = mean_features
                
                # Use numpy operations for efficient quantization
                features_quantized = (features >= threshold).astype(np.uint8)
                
            elif self.quantization_method == "median_threshold":
                # Use median as threshold (alternative method)
                median_features = np.median(features)
                threshold = median_features
                
                features_quantized = (features >= threshold).astype(np.uint8)
                
            elif self.quantization_method == "zero_threshold":
                # Use zero as threshold (simple method)
                threshold = 0.0
                
                features_quantized = (features >= threshold).astype(np.uint8)
                
            else:
                print(f"Unknown quantization method: {self.quantization_method}")
                return None
            
            print(f"Quantization: threshold={threshold:.6f}, mean={np.mean(features):.6f}")
            print(f"Binary distribution: {np.bincount(features_quantized)}")
            
            return features_quantized
            
        except Exception as e:
            print(f"Error in feature quantization: {e}")
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
            
            print(f"Processing {num_input_items} feature vector(s)")
            
            for i in range(num_input_items):
                # Get the current feature vector
                features = in0[i]
                
                # Quantize features
                quantized_features = self.feature_quantization(features)

                out0[i] = quantized_features
                print(f"✓ Quantized features: {len(quantized_features)} bits")

            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            return 0


if __name__ == "__main__":
    # Test the feature quantization block
    print("Testing feature quantization block...")
    
    # Create test data (mock features)
    test_features = np.random.randn(512).astype(np.float32)
    
    # Create block instance
    block = feature_quantization_block(quantization_method="mean_threshold")
    
    # Test feature quantization
    quantized = block.feature_quantization(test_features)
    if quantized is not None:
        print(f"✓ Test successful! Quantized shape: {quantized.shape}")
        print(f"  Binary distribution: {np.bincount(quantized)}")
    else:
        print("✗ Test failed!")
