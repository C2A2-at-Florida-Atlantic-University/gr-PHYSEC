#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParityGenerator GNU Radio Flowgraph
Generates Reed-Solomon parity bits for error correction.
"""

import numpy as np
from gnuradio import gr, blocks
from gnuradio import PHYSEC
import tempfile
import os

class ParityGenerator(gr.top_block):
    """
    GNU Radio flowgraph for Reed-Solomon parity generation
    """
    
    def __init__(self, binary_key, n=255, k=128):
        """
        Initialize parity generator flowgraph
        
        Args:
            binary_key: Binary key data (bytes)
            n: Reed-Solomon codeword length
            k: Reed-Solomon message length
        """
        gr.top_block.__init__(self, "Parity Generator")
        
        # Parameters
        self.binary_key = binary_key
        self.n = n
        self.k = k
        self.s = n - k  # Number of parity symbols
        self.key_length = len(binary_key)
        
        # Create temporary output file for parity bits
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        self.temp_filename = self.temp_file.name
        self.temp_file.close()
        
        # Build the flowgraph
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Build the parity generation flowgraph"""
        
        # Source: Binary key as vector
        self.vector_source = blocks.vector_source_b(
            list(self.binary_key),
            repeat=False,
            vlen=self.key_length,
            tags=[]
        )
        
        # PHYSEC Parity Generation Block
        self.parity_block = PHYSEC.parity_generation_block(self.n, self.k, self.key_length)
        
        # File sink for parity bits
        self.file_sink = blocks.file_sink(
            gr.sizeof_char * self.s,  # Size for parity symbols
            self.temp_filename,
            False
        )
        
        # Vector sink for accessing parity data
        self.vector_sink = blocks.vector_sink_b(self.s)
        
        # Connections
        self.connect((self.vector_source, 0), (self.parity_block, 0))
        self.connect((self.parity_block, 0), (self.file_sink, 0))
        self.connect((self.parity_block, 0), (self.vector_sink, 0))
        
    def get_parity_bits(self):
        """
        Get the generated parity bits
        
        Returns:
            bytes: Reed-Solomon parity bits or None
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
            print(f"Error reading parity bits: {e}")
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


def create_parity_generator(binary_key, n=255, k=128):
    """
    Factory function to create a ParityGenerator flowgraph
    
    Args:
        binary_key: Binary key data (bytes)
        n: Reed-Solomon codeword length
        k: Reed-Solomon message length
        
    Returns:
        ParityGenerator: Configured flowgraph instance
    """
    return ParityGenerator(binary_key, n, k)


if __name__ == '__main__':
    # Test the ParityGenerator flowgraph
    print("Testing ParityGenerator flowgraph...")
    
    # Generate test binary key
    test_key = bytes(np.random.randint(0, 256, 512))
    
    # Create parity generator
    generator = create_parity_generator(test_key)
    
    try:
        print("Starting parity generation...")
        generator.start()
        generator.wait()
        
        # Get parity bits
        parity_bits = generator.get_parity_bits()
        
        if parity_bits:
            print(f"✅ Parity generation complete!")
            print(f"   Input key length: {len(test_key)} bytes")
            print(f"   Parity bits length: {len(parity_bits)} bytes")
            print(f"   Reed-Solomon code: ({generator.n}, {generator.k})")
            print(f"   Code rate: {generator.k/generator.n:.3f}")
        else:
            print("❌ No parity bits generated")
        
    except Exception as e:
        print(f"❌ Parity generation failed: {e}")
    finally:
        generator.stop()
        generator.cleanup()
        print("🧹 Cleanup complete")
