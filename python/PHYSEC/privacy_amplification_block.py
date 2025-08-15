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
import hashlib

class privacy_amplification_block(gr.sync_block):
    """
    Privacy amplification block using SHA3-512 hashing.
    
    This block takes reconciled features and performs privacy amplification using the same
    logic as test_channel_fingerprinting_framework_onnx.py
    """
    
    def __init__(self, hash_algorithm="sha3_512"):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Privacy Amplification Block",
            in_sig=[(np.uint8, 128)],  # Binary key input
            out_sig=[(np.uint8, 128)]  # Final key output (128 bytes = 1024 bits)
        )
        
        # Store parameters
        self.hash_algorithm = hash_algorithm
        
        print(f"PHYSEC Privacy Amplification Block initialized:")
        print(f"  Hash Algorithm: {hash_algorithm}")
        print(f"  Input Size: 512 (binary key)")
        print(f"  Output Size: 128 (final key bytes)")
    
    def privacy_amplification(self, data):
        """
        Perform privacy amplification using SHA3-512 hashing.
        
        Args:
            data: Binary array or string data
            
        Returns:
            Final key as bytes array
        """
        try:            
            # Encode the string to bytes
            # Convert the binary array to a unicode string
            data_unicode = ''.join(chr(bit) for bit in data)
            encoded_str = data_unicode.encode()
            
            # Create hash object based on algorithm
            if self.hash_algorithm == "sha3_512":
                hash_obj = hashlib.new("sha3_512", encoded_str)
            elif self.hash_algorithm == "sha3_256":
                hash_obj = hashlib.new("sha3_512", encoded_str)
            elif self.hash_algorithm == "sha256":
                hash_obj = hashlib.new("sha256", encoded_str)
            else:
                print(f"Unknown hash algorithm: {self.hash_algorithm}")
                return None
            
            # Generate hash
            key_hex = hash_obj.hexdigest()
            # key_bytes = hash_obj.digest()  # Get raw bytes
            
            print(f"Privacy amplification: input_bits={len(data)}, output_bits={len(key_hex)}")
            print(f"Key hex: {key_hex}")
            print(f"Key hex length: {len(key_hex)}")
            print(f"Key hex data type: {type(key_hex)}")
            # Convert the hex string to a binary array
            key_binary = np.array([ord(char) for char in key_hex], dtype=np.uint8)
            print(f"Key binary: {key_binary}")
            print(f"Key binary length: {len(key_binary)}")
            print(f"Key binary data type: {type(key_binary)}")
            return key_binary
            
        except Exception as e:
            print(f"Error in privacy amplification: {e}")
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
            
            print(f"Processing {num_input_items} binary key(s)")
            
            for i in range(num_input_items):
                # Get the current binary key
                print(f"Input data: {in0[i]}")
                binary_key = in0[i]
                print(f"Binary key: {binary_key}")
                print(f"Binary key length: {len(binary_key)}")
                print(f"Binary key data type: {type(binary_key)}")
                # Perform privacy amplification
                final_key_bytes = self.privacy_amplification(binary_key)
                
                out0[i] = final_key_bytes
                print(f"✓ Generated final key: {len(final_key_bytes)} bytes")

            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            return 0


if __name__ == "__main__":
    # Test the privacy amplification block
    print("Testing privacy amplification block...")
    
    # Create test data (mock binary array)
    test_data = np.random.randint(0, 2, (512,), dtype=np.uint8)
    
    # Create block instance
    block = privacy_amplification_block(hash_algorithm="sha3_512")
    
    # Test privacy amplification
    final_key = block.privacy_amplification(test_data)
    if final_key is not None:
        print(f"✓ Test successful! Final key length: {len(final_key)} bytes")
        print(f"  Key hex: {final_key.hex()[:32]}...")
    else:
        print("✗ Test failed!")
