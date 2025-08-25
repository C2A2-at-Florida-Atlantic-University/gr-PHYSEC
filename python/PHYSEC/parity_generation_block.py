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
import unireedsolomon as rs

class parity_generation_block(gr.sync_block):
    """
    Parity generation block using Reed-Solomon coding.
    
    This block takes binary keys and generates parity bits for reconciliation,
    matching the test_channel_fingerprinting_framework_onnx.py logic
    """
    
    def __init__(self, n=256, k=128, key_length=512):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Parity Generation Block",
            in_sig=[(np.uint8, key_length)],  # Binary key input
            # Outputs: parity bits (n-k) and key hex (k)
            out_sig=[(np.uint8, n-k), (np.uint8, k)]
        )
        
        # Store parameters
        self.n = n  # Total codeword length
        self.k = k  # Message length
        self.s = n - k  # Parity length
        self.key_length = key_length
        # Initialize Reed-Solomon coder
        self.coder = rs.RSCoder(n, k)
        
        print(f"PHYSEC Parity Generation Block initialized:")
        print(f"  RS Code: ({n}, {k}) - {self.s} parity symbols")
        print(f"  Code Rate: {k/n:.3f}")
        print(f"  Input Size: {self.key_length} (binary key)")
        print(f"  Output Size: {self.s} (parity bits)")
    
    def arr2str(self, arr):
        """
        Convert binary array to string, matching the test file implementation.
        """
        str_arr = ''
        for i in arr:
            str_arr += str(i)
        return str_arr
    
    def generate_parity(self, binary_key):
        """
        Generate parity bits for a binary key using Reed-Solomon coding.
        
        Args:
            binary_key: Binary array (0s and 1s)
            
        Returns:
            Parity bits array or None if error
        """
        try:            
            # Convert binary array to hex string
            
            # Convert to hex (this will be truncated to fit in k symbols)
            # For RS(255, 128), we can handle up to 128 hex characters
            # 512 bits = 64 hex characters, which fits within k=128
            strKey = self.arr2str(binary_key)   
            
            hex_str = hex(int(strKey, 2))
            hex_str = str(hex_str[2:])  # Remove '0x' prefix
            # Encode using Reed-Solomon
            encoded = self.coder.encode(hex_str)
            parity_symbols = encoded[self.k:]  # Extract parity symbols
            print(f"Generated parity: input_bits={len(binary_key)}, parity_bits={len(parity_symbols)}")
            print(f"  RS Code: ({self.n}, {self.k})")
            
            # Convert parity symbols (unicode string) to expected output np.uint8
            parity_symbols_list = [ord(char) for char in parity_symbols]
            parity_symbols_uint8 = np.array(parity_symbols_list, dtype=np.uint8)
            # Also provide key hex as uint8 (ASCII) length k
            key_hex_uint8 = np.array([ord(c) for c in hex_str], dtype=np.uint8)
            
            return parity_symbols_uint8, key_hex_uint8
            
        except Exception as e:
            print(f"Error in parity generation: {e}")
            import traceback
            traceback.print_exc()
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
            out1 = output_items[1]
            num_input_items = len(in0)
            
            print(f"Processing {num_input_items} binary key(s)")
            
            for i in range(num_input_items):
                # Get the current binary key
                binary_key = in0[i]
                
                # Generate parity bits and key hex
                result = self.generate_parity(binary_key)
                
                if result is not None:
                    parity_bits, key_hex_uint8 = result
                    out0[i] = parity_bits
                    out1[i] = key_hex_uint8
                    print(f"✓ Generated parity bits: {len(parity_bits)} bits; key hex: {len(key_hex_uint8)} chars")
                else:
                    print(f"✗ Failed to generate parity bits for key {i}")
                    # Fill with zeros if generation failed
                    out0[i] = np.zeros(self.s, dtype=np.uint8)
                    out1[i] = np.zeros(self.k, dtype=np.uint8)
                
            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            import traceback
            traceback.print_exc()
            return 0


if __name__ == "__main__":
    # Test the parity generation block
    print("Testing parity generation block...")
    
    # Create test data (mock binary key)
    test_binary_key = np.random.randint(0, 2, (512,), dtype=np.uint8)
    
    # Create block instance
    block = parity_generation_block(n=255, k=128, key_length=512)
    
    # Test parity generation
    parity_bits = block.generate_parity(test_binary_key)
    if parity_bits is not None:
        print(f"✓ Test successful! Parity bits length: {len(parity_bits)}")
        print(f"  Parity bits: {parity_bits[:10]}...")
    else:
        print("✗ Test failed!")
