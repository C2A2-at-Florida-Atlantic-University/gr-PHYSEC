#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-PHYSEC author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""
Test script for the decoupled Physical Key Generation Framework.

This script demonstrates how the six separate blocks work together:
1. Spectrogram Creation
2. Feature Extraction  
3. Feature Quantization
4. Parity Generation
5. Reconciliation
6. Privacy Amplification

Each block can be tested individually or as part of the complete pipeline.
"""

import sys
import os
import numpy as np
import time
import h5py

# Include the AI path at /workspace/siwn/siwn-node/AI
ai_dir = '/workspace/siwn/siwn-node/AI'
if ai_dir not in sys.path:
    sys.path.append(ai_dir)

# Import the decoupled blocks
from spectrogram_block import spectrogram_block
from feature_extraction_block import feature_extraction_block
from feature_quantization_block import feature_quantization_block
from parity_generation_block import parity_generation_block
from reconciliation_block import reconciliation_block
from privacy_amplification_block import privacy_amplification_block

def test_individual_blocks():
    """Test each block individually with mock data."""
    print("=" * 80)
    print("TESTING INDIVIDUAL BLOCKS")
    print("=" * 80)
    
    # Test 1: Spectrogram Block
    print("\n1. Testing Spectrogram Block")
    print("-" * 40)
    
    # Create mock IQ data
    test_iq = np.random.randn(8192) + 1j * np.random.randn(8192)
    test_iq = test_iq.astype(np.complex64)
    
    # Create spectrogram block
    spec_block = spectrogram_block(fft_window=512, vector_size=8192)
    
    # Test spectrogram creation
    spectrogram = spec_block.create_spectrogram(test_iq)
    if spectrogram is not None:
        print(f"✓ Spectrogram created: {spectrogram.shape}")
    else:
        print("✗ Spectrogram creation failed")
        return False
    
    # Test 2: Feature Extraction Block
    print("\n2. Testing Feature Extraction Block")
    print("-" * 40)
    
    # Create feature extraction block
    model_path = "/workspace/data/gr-PHYSEC/models/QExtractor.onnx"
    feat_block = feature_extraction_block(model_path)
    
    # Test feature extraction
    features = feat_block.extract_features(spectrogram)
    if features is not None:
        print(f"✓ Features extracted: {features.shape}")
    else:
        print("✗ Feature extraction failed")
        return False
    
    # Test 3: Feature Quantization Block
    print("\n3. Testing Feature Quantization Block")
    print("-" * 40)
    
    # Create feature quantization block
    quant_block = feature_quantization_block(quantization_method="mean_threshold")
    
    # Test feature quantization
    quantized = quant_block.feature_quantization(features)
    if quantized is not None:
        print(f"✓ Features quantized: {quantized.shape}")
        print(f"  Binary distribution: {np.bincount(quantized)}")
    else:
        print("✗ Feature quantization failed")
        return False
    
    # Test 4: Parity Generation Block
    print("\n4. Testing Parity Generation Block")
    print("-" * 40)
    k = int(512/4)
    n = int(k+(k/1)-1)
    # Create parity generation block
    parity_block = parity_generation_block(n=n, k=k, key_length=512)
    
    # Test parity generation
    parity_bits = parity_block.generate_parity(quantized)
    if parity_bits is not None:
        print(f"✓ Parity bits generated: {parity_bits}")
        print(f"  Parity bits length: {len(parity_bits)}")
        # print(f"  Parity bits: {parity_bits[:10]}...")
    else:
        print("✗ Parity generation failed")
        return False
    
    # Test 5: Reconciliation Block
    print("\n5. Testing Reconciliation Block")
    print("-" * 40)
    
    # Create reconciliation block
    recon_block = reconciliation_block(n=n, k=k, key_length=512)
    
    # Test reconciliation
    reconciled_key = recon_block.reconcile(quantized, parity_bits)
    if reconciled_key is not None:
        print(f"✓ Reconciliation completed: {reconciled_key}")
        print(f"  Original key: {quantized[:]}")
        print(f"  Reconciled key: {reconciled_key[:]}")
    else:
        print("✗ Reconciliation failed")
        return False
    
    # Test 6: Privacy Amplification Block
    print("\n6. Testing Privacy Amplification Block")
    print("-" * 40)
    
    # Create privacy amplification block
    priv_block = privacy_amplification_block(hash_algorithm="sha3_512")
    
    # Test privacy amplification
    final_key = priv_block.privacy_amplification(reconciled_key)
    if final_key is not None:
        print(f"✓ Final key generated: {len(final_key)} bytes")
        print(f"  Key hex: {final_key}")
        print(f"  Key hex length: {len(final_key)}")
        print(f"  Key hex data type: {type(final_key)}")
    else:
        print("✗ Privacy amplification failed")
        return False
    
    print("\n✓ All individual block tests passed!")
    return True

def test_complete_pipeline():
    """Test the complete pipeline with real data from the dataset."""
    print("\n" + "="*80)
    print("TESTING COMPLETE PIPELINE")
    print("="*80)
    
    try:
        from dataset_preparation import LoadDatasetChannels
        
        # Load test data
        print("Loading test data...")
        PHYSEC_dir = '/workspace/data/gr-PHYSEC/'
        dataset_path = PHYSEC_dir + 'datasets/'
        dataset = dataset_path + 'Dataset_Channels_sinusoid_dev_871_1690302750.hdf5'
        
        LoadDatasetObj = LoadDatasetChannels()
        data, labels = LoadDatasetObj.load_iq_samples(dataset)
        
        # Use first few samples for testing
        test_data = data[:4]
        print(f"Test data shape: {test_data.shape}")
        
        # Initialize all blocks
        print("\nInitializing blocks...")
        
        spec_block = spectrogram_block(fft_window=512, vector_size=len(test_data[0]))
        model_path = "/workspace/data/gr-PHYSEC/models/QExtractor.onnx"
        feat_block = feature_extraction_block(model_path)
        quant_block = feature_quantization_block(quantization_method="mean_threshold")
        k = int(512/4)
        n = int(k+(k/1)-1)
        parity_block = parity_generation_block(n=n, k=k, key_length=512)
        recon_block = reconciliation_block(n=n, k=k, key_length=512)
        priv_block = privacy_amplification_block(hash_algorithm="sha3_512")
        
        # Process each sample through the complete pipeline
        print("\nProcessing samples through complete pipeline...")
        
        all_quantized = []
        all_parity_bits = []
        all_reconciled_keys = []
        all_final_keys = []
        
        for i, iq_sample in enumerate(test_data):
            print(f"\n--- Processing Sample {i+1} ---")
            
            # Step 1: Create spectrogram
            print("1. Creating spectrogram...")
            spectrogram = spec_block.create_spectrogram(iq_sample)
            print(f"Spectrogram shape: {spectrogram.shape}")
            if spectrogram is None:
                print(f"✗ Failed to create spectrogram for sample {i+1}")
                continue
            
            # Step 2: Extract features
            print("2. Extracting features...")
            features = feat_block.extract_features(spectrogram)
            print(f"Features shape: {features.shape}")
            if features is None:
                print(f"✗ Failed to extract features for sample {i+1}")
                continue
            
            # Step 3: Quantize features
            print("3. Quantizing features...")
            quantized = quant_block.feature_quantization(features)
            print(f"Quantized shape: {quantized.shape}")
            if quantized is None:
                print(f"✗ Failed to quantize features for sample {i+1}")
                continue
            
            all_quantized.append(quantized)
            
            # Step 4: Generate parity bits
            print("4. Generating parity bits...")
            parity_bits = parity_block.generate_parity(quantized)
            print(f"Parity bits shape: {len(parity_bits)}")
            if parity_bits is None:
                print(f"✗ Failed to generate parity bits for sample {i+1}")
                continue
            
            all_parity_bits.append(parity_bits)
            
            # Step 5: Perform reconciliation
            print("5. Performing reconciliation...")
            reconciled_key = recon_block.reconcile(quantized, parity_bits)
            print(f"Reconciled key shape: {len(reconciled_key)}")
            if reconciled_key is None:
                print(f"✗ Failed to perform reconciliation for sample {i+1}")
                continue
            
            all_reconciled_keys.append(reconciled_key)
            
            # Step 6: Privacy amplification
            print("6. Performing privacy amplification...")
            final_key = priv_block.privacy_amplification(reconciled_key)
            print(f"Final key shape: {len(final_key)}")
            if final_key is None:
                print(f"✗ Failed to perform privacy amplification for sample {i+1}")
                continue
            
            all_final_keys.append(final_key)
            print(f"✓ Sample {i+1} completed successfully")
        
        # Save results
        print("\n--- Saving Results ---")
        output_file = dataset_path + 'Dataset_Channels_complete_pipeline_stream_test.hdf5'
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('quantized_features', data=np.array(all_quantized))
            # f.create_dataset('parity_bits', data=np.array(all_parity_bits))
            # f.create_dataset('reconciled_keys', data=np.array(all_reconciled_keys))
            # f.create_dataset('final_keys', data=np.array(all_final_keys))
            # f.create_dataset('metadata', data=str({
            #     'source': 'complete_pipeline_stream_test',
            #     'num_samples': len(all_quantized),
            #     'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            #     'fft_window': 512,
            #     'rs_code': '(255, 128)',
            #     'hash_algorithm': 'sha3_512',
            #     'block_type': 'stream_based'
            # }))
        
        print(f"✓ Results saved to: {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        print(f"\n✓ Complete pipeline test successful!")
        print(f"  Processed {len(all_quantized)} samples")
        print(f"  Generated {len(all_final_keys)} final keys")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in complete pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stream_connectivity():
    """Test the stream connectivity between blocks."""
    print("\n" + "="*80)
    print("TESTING STREAM CONNECTIVITY")
    print("="*80)
    
    try:
        from dataset_preparation import LoadDatasetChannels
        # Create mock data
        # test_iq = np.load('/workspace/data/gr-PHYSEC/datasets/Dataset_Channels_sinusoid_dev_871_1690302750.hdf5')
        dataset_path = '/workspace/data/gr-PHYSEC/datasets/'
        dataset = dataset_path + 'Dataset_Channels_sinusoid_dev_871_1690302750.hdf5'
        LoadDatasetObj = LoadDatasetChannels()
        data, labels = LoadDatasetObj.load_iq_samples(dataset)
        number_of_test_samples = 10
        quantized_features = []
        fft_window = 512
        iq_length = 8192
        spec_block = spectrogram_block(fft_window=fft_window, vector_size=iq_length)
        model_path = "/workspace/data/gr-PHYSEC/models/QExtractor.onnx"
        feat_block = feature_extraction_block(model_path)
        quant_block = feature_quantization_block(quantization_method="mean_threshold")
        key_length = 512
        k = int(key_length/4)
        n = int(k+(k/1)-1)
        print(f"K: {k}")
        print(f"N: {n}")
        print(f"Key length: {key_length}")
        parity_block = parity_generation_block(n=n, k=k, key_length=key_length)
        recon_block = reconciliation_block(n=n, k=k, key_length=key_length)
        priv_block = privacy_amplification_block(hash_algorithm="sha3_512")
        # Simulate stream processing
        print("Simulating stream processing...")
        for test_iq in data[:number_of_test_samples]:
            # iq_length = len(test_iq)
            # test_iq = test_iq.astype(np.complex64)
            # Input data
            input_items = [[test_iq]]
            output_items = [np.zeros((1, 204, 31), dtype=np.float32)]
            
            # Process through spectrogram block
            print("1. Spectrogram block...")
            items_processed = spec_block.work(input_items, output_items)
            spectrogram_output = output_items[0][0]
            print(f"   Output shape: {spectrogram_output.shape}")
            
            # Process through feature extraction block
            print("2. Feature extraction block...")
            feat_input = [[spectrogram_output]]
            feat_output = [np.zeros((1, key_length), dtype=np.float32)]
            items_processed = feat_block.work(feat_input, feat_output)
            features_output = feat_output[0][0]
            print(f"   Output shape: {features_output.shape}")
            
            # Process through quantization block
            print("3. Feature quantization block...")
            quant_input = [[features_output]]
            quant_output = [np.zeros((1, key_length), dtype=np.uint8)]
            items_processed = quant_block.work(quant_input, quant_output)
            quantized_output = quant_output[0][0]
            quantized_features.append(quantized_output)
            print(f"   Output shape: {quantized_output.shape}")
            
            # Process through parity generation block
            print("4. Parity generation block...")
            parity_input = [[quantized_output]]
            # Create a list to hold the string output from parity generation
            parity_output = [[''] * (n-k)]  # List of strings, not numpy array
            items_processed = parity_block.work(parity_input, parity_output)
            parity_output_data = parity_output[0][0]
            print(f"   Parity output data: {parity_output_data}")
            print(f"   Parity output data length: {len(parity_output_data)}")
            print(f"   Parity output data type: {type(parity_output_data)}")
            print(f"   Output shape: {len(parity_output_data)}")
            
            # Process through reconciliation block
            print("5. Reconciliation block...")
            recon_input = [[quantized_output], [parity_output_data]]
            # Create a list to hold the string output from reconciliation
            recon_output = [[''] * 128]  # List of strings, not numpy array
            items_processed = recon_block.work(recon_input, recon_output)
            reconciled_output = recon_output[0][0]
            print(f"   Output shape: {len(reconciled_output)}")
            print(f"   Reconciliation data: {reconciled_output}")
            print(f"   Reconciliation data type: {type(reconciled_output)}")
            
            # Process through privacy amplification block
            print("6. Privacy amplification block...")
            # print(f"Reconciled output: {reconciled_output}")
            priv_input = [[reconciled_output]]
            # priv_output = [np.zeros((1, 128), dtype=np.uint8)]
            priv_output = [[''] * 128]
            items_processed = priv_block.work(priv_input, priv_output)
            final_key_output = priv_output[0][0]
            print(f"   Key output shape: {len(final_key_output)}")
            print(f"   Key output data: {final_key_output}")
            print(f"   Key output data type: {type(final_key_output)}")
            
            print("\n✓ Stream connectivity test successful!")
            print(f"  Final key generated: {len(final_key_output)} bytes")
        
        # Get the quantized features from /workspace/data/gr-PHYSEC/datasets/Dataset_Channels_quantized_TF38Model_dev_871_1690302750.hdf5
        quantized_features_path = dataset_path + 'Dataset_Channels_quantized_TF38Model_dev_871_1690302750.hdf5'
        with h5py.File(quantized_features_path, 'r') as f:
            quantized_features_TF38 = f['quantized_keys'][:]
        print(f"Quantized features shape: {quantized_features_TF38.shape}")
        print(f"Quantized features: {quantized_features_TF38}")
        print(f"Quantized features data type: {type(quantized_features_TF38)}")
        # Compare the quantized features with the quantized features from the stream connectivity test
        if np.array_equal(quantized_features, quantized_features_TF38[:number_of_test_samples]):
            print("✓ Quantized features match")
        else:
            print("✗ Quantized features do not match")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in stream connectivity test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("PHYSICAL KEY GENERATION FRAMEWORK - COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Test individual blocks
    if not test_individual_blocks():
        print("\n✗ Individual block tests failed. Stopping.")
        return
    
    # Test stream connectivity
    if not test_stream_connectivity():
        print("\n✗ Stream connectivity test failed. Stopping.")
        return
    
    # Test complete pipeline
    if not test_complete_pipeline():
        print("\n✗ Complete pipeline test failed.")
        return
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*80)
    print("\nThe decoupled Physical Key Generation Framework is working correctly.")
    print("Each block can now be used independently in GNU Radio flowgraphs.")
    print("Blocks use stream inputs/outputs for easy integration.")

if __name__ == "__main__":
    main()
