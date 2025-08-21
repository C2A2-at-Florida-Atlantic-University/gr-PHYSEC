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

class reconciliation_block(gr.sync_block):
    """
    Reconciliation block using Reed-Solomon coding.
    
    This block takes a binary key and parity bits as input and attempts to
    perform reconciliation, outputting the reconciled key and success status.
    """
    
    def __init__(self, n=255, k=128, key_length=512):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Reconciliation Block",
            in_sig=[(np.uint8, key_length), (np.uint8, n-k)],  # Binary key + parity bits input
            out_sig=[(np.uint8, k), np.uint8]  # Reconciled key output + success flag
        )
        
        # Store parameters
        self.n = n  # Total codeword length
        self.k = k  # Message length
        self.s = n - k  # Parity length
        self.key_length = key_length
        # Initialize Reed-Solomon coder
        self.coder = rs.RSCoder(n, k)
        
        print(f"PHYSEC Reconciliation Block initialized:")
        print(f"  RS Code: ({n}, {k}) - {self.s} parity symbols")
        print(f"  Code Rate: {k/n:.3f}")
        print(f"  Input 1 Size: {self.key_length} (binary key)")
        print(f"  Input 2 Size: {self.s} (parity bits)")
        print(f"  Output 1 Size: {self.k} (reconciled key)")
        print(f"  Output 2 Size: 1 (success flag)")
    
    def arr2str(self, arr):
        """
        Convert binary array to string, matching the test file implementation.
        """
        str_arr = ''
        for i in arr:
            str_arr += str(i)
        return str_arr
    
    def reconcile(self, binary_key, parity_bits):
        """
        Reconcile a binary key using parity bits and Reed-Solomon coding.
        
        Args:
            binary_key: Binary array (0s and 1s) to be reconciled
            parity_bits: Parity bits array for reconciliation
            
        Returns:
            Tuple: (reconciled_key, success_flag)
                - reconciled_key: Reconciled key if successful, original key if failed
                - success_flag: True if reconciliation succeeded, False otherwise
        """
        try:
            
            # Convert binary key to hex string
            binary_str = self.arr2str(binary_key)
            hex_str = hex(int(binary_str, 2))
            hex_str = str(hex_str[2:])  # Remove '0x' prefix
            
            # Convert parity bits which is a np.uint8 array to a unicode string
            parity2unicode = str(''.join(chr(bit) for bit in parity_bits))            
            # Combine key with parity bits
            combined_data = hex_str + parity2unicode
            print(f"Combined data: {len(hex_str)} + {len(parity_bits)} = {len(combined_data)} hex chars")
            
            try:
                # Attempt to decode using Reed-Solomon
                decoded = self.coder.decode(combined_data)
                reconciled_hex = decoded[0]  # Get the decoded message
                # Convert the hex string to a binary array
                # reconciled_binary = np.array([int(char) for char in reconciled_hex], dtype=np.uint8)
                reconciled_binary = np.array([ord(char) for char in reconciled_hex], dtype=np.uint8)
                
                print(f"RS decoding successful: {len(combined_data)} -> {len(decoded[0])} hex chars")
                print(f"Reconciliation successful: input_bits={len(binary_key)}, output_bits={len(reconciled_binary)}")
                return reconciled_binary, True
                
            except Exception as decode_error:
                print(f"Reconciliation failed: {decode_error}")
                # Return original key if reconciliation fails
                failed_key = np.array([ord(char) for char in hex_str], dtype=np.uint8)
                return failed_key, False
            
        except Exception as e:
            print(f"Error in reconciliation: {e}")
            import traceback
            traceback.print_exc()
            failed_key = np.array([ord(char) for char in hex_str], dtype=np.uint8)
            return failed_key, False
    
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
            in0 = input_items[0]  # Binary key
            in1 = input_items[1]  # Parity bits
            out0 = output_items[0]  # Reconciled key
            out1 = output_items[1]  # Success flag
            
            num_input_items = len(in0)
            
            print(f"Processing {num_input_items} reconciliation(s)")
            
            for i in range(num_input_items):
                # Get the current binary key and parity bits
                binary_key = in0[i]
                parity_bits = in1[i]
                
                # Perform reconciliation
                reconciled_key, success = self.reconcile(binary_key, parity_bits)

                out0[i] = reconciled_key
                out1[i] = 1 if success else 0  # Convert boolean to uint8
                print(f"✓ Reconciliation completed: {len(reconciled_key)} bits, success: {success}")
                
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            import traceback
            traceback.print_exc()
            return 0


if __name__ == "__main__":
    # Test the reconciliation block
    print("Testing reconciliation block...")
    
    # Create test data (mock binary key and parity bits)
    test_binary_key = np.random.randint(0, 2, (512,), dtype=np.uint8)
    test_parity_bits = np.random.randint(0, 2, (127,), dtype=np.uint8)  # n-k = 255-128 = 127
    
    # Create block instance
    block = reconciliation_block(n=255, k=128)
    
    # Test reconciliation
    reconciled_key, success = block.reconcile(test_binary_key, test_parity_bits)
    if success:
        print(f"✓ Test successful! Reconciliation succeeded.")
        print(f"  Reconciled key shape: {reconciled_key.shape}")
        print(f"  Original key: {test_binary_key[:10]}...")
        print(f"  Reconciled key: {reconciled_key[:10]}...")
        
        # Check how many bits changed
        bit_changes = np.sum(test_binary_key != reconciled_key)
        print(f"  Bits changed: {bit_changes}")
    else:
        print("⚠ Test returned failure flag, but key was returned anyway")
        print(f"  Returned key shape: {reconciled_key.shape}")
        print(f"  Original key: {test_binary_key[:10]}...")
        print(f"  Returned key: {reconciled_key[:10]}...")
