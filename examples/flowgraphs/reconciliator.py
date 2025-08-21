#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconciliator GNU Radio Flowgraph
Performs Reed-Solomon reconciliation for key agreement.
"""

import numpy as np
from gnuradio import gr, blocks
from gnuradio import PHYSEC
import tempfile
import os

class Reconciliator(gr.top_block):
    """
    GNU Radio flowgraph for Reed-Solomon reconciliation
    """
    
    def __init__(self, binary_key, parity_bits, n=255, k=128):
        """
        Initialize reconciliator flowgraph
        
        Args:
            binary_key: Binary key data (bytes)
            parity_bits: Reed-Solomon parity bits (bytes)
            n: Reed-Solomon codeword length
            k: Reed-Solomon message length
        """
        gr.top_block.__init__(self, "Reconciliator")
        
        # Parameters
        self.binary_key = binary_key
        self.parity_bits = parity_bits
        self.n = n
        self.k = k
        self.s = n - k  # Number of parity symbols
        self.key_length = len(binary_key)
        
        # Create temporary output files
        self.reconciled_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        self.reconciled_filename = self.reconciled_file.name
        self.reconciled_file.close()
        
        self.success_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        self.success_filename = self.success_file.name
        self.success_file.close()
        
        # Build the flowgraph
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Build the reconciliation flowgraph"""
        
        # Source 1: Binary key as vector
        self.key_source = blocks.vector_source_b(
            list(self.binary_key),
            repeat=False,
            vlen=self.key_length,
            tags=[]
        )
        
        # Source 2: Parity bits as vector
        self.parity_source = blocks.vector_source_b(
            list(self.parity_bits),
            repeat=False,
            vlen=self.s,
            tags=[]
        )
        
        # PHYSEC Reconciliation Block (with two outputs)
        self.reconciliation_block = PHYSEC.reconciliation_block(self.n, self.k, self.key_length)
        
        # File sinks for outputs
        self.reconciled_sink = blocks.file_sink(
            gr.sizeof_char * self.k,  # Reconciled key size
            self.reconciled_filename,
            False
        )
        
        self.success_sink = blocks.file_sink(
            gr.sizeof_char,  # Success flag (single byte)
            self.success_filename,
            False
        )
        
        # Vector sinks for accessing data
        self.reconciled_vector_sink = blocks.vector_sink_b(self.k)
        self.success_vector_sink = blocks.vector_sink_b(1)
        
        # Connections
        self.connect((self.key_source, 0), (self.reconciliation_block, 0))
        self.connect((self.parity_source, 0), (self.reconciliation_block, 1))
        
        # Connect outputs
        self.connect((self.reconciliation_block, 0), (self.reconciled_sink, 0))
        self.connect((self.reconciliation_block, 1), (self.success_sink, 0))
        self.connect((self.reconciliation_block, 0), (self.reconciled_vector_sink, 0))
        self.connect((self.reconciliation_block, 1), (self.success_vector_sink, 0))
        
    def get_reconciled_key(self):
        """
        Get the reconciled key
        
        Returns:
            bytes: Reconciled key or None
        """
        try:
            # Try to get from vector sink first
            data = self.reconciled_vector_sink.data()
            if data:
                return bytes(data)
                
            # Fallback: read from file
            with open(self.reconciled_filename, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading reconciled key: {e}")
            return None
    
    def get_success_flag(self):
        """Alias for get_reconciliation_success for compatibility"""
        return self.get_reconciliation_success()
            
    def get_reconciliation_success(self):
        """
        Get the reconciliation success flag
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to get from vector sink first
            data = self.success_vector_sink.data()
            if data:
                return bool(data[0])
                
            # Fallback: read from file
            with open(self.success_filename, 'rb') as f:
                success_byte = f.read(1)
                return bool(success_byte[0]) if success_byte else False
        except Exception as e:
            print(f"Error reading reconciliation success: {e}")
            return False
            
    def get_results(self):
        """
        Get both reconciled key and success flag
        
        Returns:
            tuple: (reconciled_key, success) or (None, False)
        """
        key = self.get_reconciled_key()
        success = self.get_reconciliation_success()
        return key, success
        
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.reconciled_filename):
                os.remove(self.reconciled_filename)
            if os.path.exists(self.success_filename):
                os.remove(self.success_filename)
        except Exception as e:
            print(f"Warning: Could not remove temp files: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


def create_reconciliator(binary_key, parity_bits, n=255, k=128):
    """
    Factory function to create a Reconciliator flowgraph
    
    Args:
        binary_key: Binary key data (bytes)
        parity_bits: Reed-Solomon parity bits (bytes)
        n: Reed-Solomon codeword length
        k: Reed-Solomon message length
        
    Returns:
        Reconciliator: Configured flowgraph instance
    """
    return Reconciliator(binary_key, parity_bits, n, k)


if __name__ == '__main__':
    # Test the Reconciliator flowgraph
    print("Testing Reconciliator flowgraph...")
    
    # Generate test data
    test_key = bytes(np.random.randint(0, 256, 512))
    test_parity = bytes(np.random.randint(0, 256, 127))  # s = n - k = 255 - 128 = 127
    
    # Create reconciliator
    reconciliator = create_reconciliator(test_key, test_parity)
    
    try:
        print("Starting reconciliation...")
        reconciliator.start()
        reconciliator.wait()
        
        # Get results
        reconciled_key, success = reconciliator.get_results()
        
        print(f"✅ Reconciliation complete!")
        print(f"   Input key length: {len(test_key)} bytes")
        print(f"   Parity bits length: {len(test_parity)} bytes")
        print(f"   Reconciled key length: {len(reconciled_key) if reconciled_key else 0} bytes")
        print(f"   Reconciliation success: {success}")
        print(f"   Reed-Solomon code: ({reconciliator.n}, {reconciliator.k})")
        
    except Exception as e:
        print(f"❌ Reconciliation failed: {e}")
    finally:
        reconciliator.stop()
        reconciliator.cleanup()
        print("🧹 Cleanup complete")
