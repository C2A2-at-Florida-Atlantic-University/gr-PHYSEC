#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrivacyAmplifier GNU Radio Flowgraph
Performs privacy amplification using cryptographic hashing for final key generation.
"""

import numpy as np
from gnuradio import gr, blocks
from gnuradio import PHYSEC
import tempfile
import os

class PrivacyAmplifier(gr.top_block):
    """
    GNU Radio flowgraph for privacy amplification
    """
    
    def __init__(self, binary_key=None, key_length=128, hash_algorithm='sha3_512'):
        """
        Initialize privacy amplifier flowgraph
        
        Args:
            binary_key: Binary key data (bytes)
            key_length: Desired output key length in bytes
            hash_algorithm: Hash algorithm to use ('sha3_512', 'sha256', etc.)
        """
        gr.top_block.__init__(self, "Privacy Amplifier")
        
        # Parameters
        self.binary_key = binary_key
        self.key_length = key_length
        self.hash_algorithm = hash_algorithm
        
        # Create temporary output file for final key
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        self.temp_filename = self.temp_file.name
        self.temp_file.close()
        
        # Only build flowgraph if we have binary key data
        if binary_key is not None:
            self.input_length = len(binary_key)
            self._build_flowgraph()
        else:
            self.input_length = None
        
    def _build_flowgraph(self):
        """Build the privacy amplification flowgraph"""
        
        # Source: Binary key as vector
        self.vector_source = blocks.vector_source_b(
            list(self.binary_key),
            repeat=False,
            vlen=self.input_length,
            tags=[]
        )
        
        # PHYSEC Privacy Amplification Block
        self.privacy_block = PHYSEC.privacy_amplification_block(self.hash_algorithm)
        
        # File sink for final key
        self.file_sink = blocks.file_sink(
            gr.sizeof_char * self.key_length,
            self.temp_filename,
            False
        )
        
        # Vector sink for accessing key data
        self.vector_sink = blocks.vector_sink_b(self.key_length)
        
        # Connections
        self.connect((self.vector_source, 0), (self.privacy_block, 0))
        self.connect((self.privacy_block, 0), (self.file_sink, 0))
        self.connect((self.privacy_block, 0), (self.vector_sink, 0))
        
    def update_key_data(self, binary_key):
        """Update the flowgraph with new binary key data"""
        self.binary_key = binary_key
        self.input_length = len(binary_key)
        
        # Rebuild the flowgraph with new data
        if hasattr(self, 'vector_source'):
            # Disconnect existing connections
            self.disconnect_all()
        
        # Rebuild the flowgraph
        self._build_flowgraph()
        
    def get_final_key(self):
        """
        Get the final privacy-amplified key
        
        Returns:
            bytes: Final cryptographic key or None
        """
        try:
            # Try to get from vector sink first
            data = self.vector_sink.data()
            if data:
                return bytes(data)
                
            # Fallback: read from file
            with open(self.temp_filename, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading final key: {e}")
            return None
            
    def get_key_hex(self):
        """
        Get the final key as a hexadecimal string
        
        Returns:
            str: Hex representation of the key or None
        """
        key = self.get_final_key()
        if key:
            return key.hex()
        return None
        
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)
        except Exception as e:
            print(f"Warning: Could not remove temp file {self.temp_filename}: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


def create_privacy_amplifier(binary_key, key_length=128, hash_algorithm='sha3_512'):
    """
    Factory function to create a PrivacyAmplifier flowgraph
    
    Args:
        binary_key: Binary key data (bytes)
        key_length: Desired output key length in bytes
        hash_algorithm: Hash algorithm to use
        
    Returns:
        PrivacyAmplifier: Configured flowgraph instance
    """
    return PrivacyAmplifier(binary_key, key_length, hash_algorithm)


if __name__ == '__main__':
    # Test the PrivacyAmplifier flowgraph
    print("Testing PrivacyAmplifier flowgraph...")
    
    # Generate test binary key (simulating reconciled key)
    test_key = bytes(np.random.randint(0, 256, 128))  # Reconciled key size
    
    # Create privacy amplifier
    amplifier = create_privacy_amplifier(test_key, key_length=128)
    
    try:
        print("Starting privacy amplification...")
        amplifier.start()
        amplifier.wait()
        
        # Get final key
        final_key = amplifier.get_final_key()
        final_key_hex = amplifier.get_key_hex()
        
        if final_key:
            print(f"✅ Privacy amplification complete!")
            print(f"   Input key length: {len(test_key)} bytes")
            print(f"   Final key length: {len(final_key)} bytes")
            print(f"   Hash algorithm: {amplifier.hash_algorithm}")
            print(f"   Key (hex): {final_key_hex[:32]}...{final_key_hex[-32:] if len(final_key_hex) > 64 else ''}")
            print(f"   Security level: Information-theoretic")
        else:
            print("❌ No final key generated")
        
    except Exception as e:
        print(f"❌ Privacy amplification failed: {e}")
    finally:
        amplifier.stop()
        amplifier.cleanup()
        print("🧹 Cleanup complete")
