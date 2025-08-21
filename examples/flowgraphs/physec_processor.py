#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysecProcessor GNU Radio Flowgraph
Processes IQ samples through PHYSEC spectrogram, feature extraction, and quantization blocks.
"""

import numpy as np
from gnuradio import gr, blocks
from gnuradio import PHYSEC
import tempfile
import os

class PhysecProcessor(gr.top_block):
    """
    GNU Radio flowgraph for PHYSEC signal processing pipeline
    """
    
    def __init__(self, samples, fft_window=512, vector_size=8192):
        """
        Initialize PHYSEC processor flowgraph
        
        Args:
            samples: Complex IQ samples to process
            fft_window: FFT window size for spectrogram
            vector_size: Input vector size (should match samples length)
        """
        gr.top_block.__init__(self, "PHYSEC Processor")
        
        # Parameters
        self.fft_window = fft_window
        self.vector_size = vector_size
        self.samples = samples
        
        # Create temporary output file for quantized features
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        self.temp_filename = self.temp_file.name
        self.temp_file.close()
        
        # Build the flowgraph
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Build the PHYSEC processing flowgraph"""
        
        # Source: Vector of IQ samples
        self.vector_source = blocks.vector_source_c(
            self.samples, 
            repeat=False, 
            vlen=1, 
            tags=[]
        )
        
        # Stream to Vector converter
        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex, 
            self.vector_size
        )
        
        # PHYSEC Spectrogram Block
        self.spectrogram_block = PHYSEC.spectrogram_block(
            self.fft_window, 
            self.vector_size
        )
        
        # PHYSEC Feature Extraction Block
        self.feature_extraction_block = PHYSEC.feature_extraction_block(
            '/workspace/data/gr-PHYSEC/models/QExtractor.onnx'
        )
        
        # PHYSEC Quantization Block
        self.quantization_block = PHYSEC.feature_quantization_block('mean_threshold')
        
        # Vector to Stream converter for spectrogram
        # The spectrogram block output is 204*31 = 6324 floats  
        spec_len = 6324  # Fixed size based on PHYSEC spectrogram block output
        self.vector_to_stream_spec = blocks.vector_to_stream(
            gr.sizeof_float, 
            spec_len
        )
        
        # File sink for quantized features
        self.file_sink = blocks.file_sink(
            gr.sizeof_char * 512,  # Match the quantization block output size
            self.temp_filename,
            False
        )
        
        # Vector sink for spectrogram data (for visualization)
        self.spectrogram_sink = blocks.vector_sink_f()
        
        # Connections
        self.connect((self.vector_source, 0), (self.stream_to_vector, 0))
        self.connect((self.stream_to_vector, 0), (self.spectrogram_block, 0))
        self.connect((self.spectrogram_block, 0), (self.feature_extraction_block, 0))
        self.connect((self.feature_extraction_block, 0), (self.quantization_block, 0))
        self.connect((self.quantization_block, 0), (self.file_sink, 0))
        
        # Additional connection for spectrogram visualization
        self.connect((self.spectrogram_block, 0), (self.vector_to_stream_spec, 0))
        self.connect((self.vector_to_stream_spec, 0), (self.spectrogram_sink, 0))
        
    def get_quantized_bits(self):
        """
        Read the quantized bits from the temporary file
        
        Returns:
            bytes: Quantized feature bits
        """
        import os
        import time
        
        # Wait a bit for file to be written
        time.sleep(0.1)
        
        try:
            # Check if file exists and has data
            if not os.path.exists(self.temp_filename):
                print(f"Error: Quantized bits file does not exist: {self.temp_filename}")
                return None
                
            file_size = os.path.getsize(self.temp_filename)
            if file_size == 0:
                print(f"Error: Quantized bits file is empty: {self.temp_filename}")
                return None
                
            with open(self.temp_filename, 'rb') as f:
                data = f.read()
                return data
        except Exception as e:
            print(f"Error reading quantized bits: {e}")
            return None
            
    def get_spectrogram_data(self, expected_size=6324):
        """
        Get the spectrogram data for visualization
        
        Args:
            expected_size: Expected spectrogram data size
            
        Returns:
            numpy.ndarray: Reshaped spectrogram data (204, 31) or None
        """
        try:
            spec_data = self.spectrogram_sink.data()
            if len(spec_data) >= expected_size:
                return np.array(spec_data[:expected_size]).reshape(204, 31)
            else:
                print(f"Warning: Spectrogram data size {len(spec_data)} < expected {expected_size}")
                return None
        except Exception as e:
            print(f"Error getting spectrogram data: {e}")
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


def create_physec_processor(samples, fft_window=512, vector_size=8192):
    """
    Factory function to create a PhysecProcessor flowgraph
    
    Args:
        samples: Complex IQ samples to process
        fft_window: FFT window size for spectrogram 
        vector_size: Input vector size
        
    Returns:
        PhysecProcessor: Configured flowgraph instance
    """
    return PhysecProcessor(samples, fft_window, vector_size)


if __name__ == '__main__':
    # Test the PhysecProcessor with dummy data
    print("Testing PhysecProcessor flowgraph...")
    
    # Generate test IQ samples
    test_samples = np.random.normal(0, 0.1, 8192) + 1j * np.random.normal(0, 0.1, 8192)
    
    # Create and run processor
    processor = create_physec_processor(test_samples)
    
    try:
        print("Starting PHYSEC processing...")
        processor.start()
        processor.wait()
        
        # Get results
        quantized_bits = processor.get_quantized_bits()
        spectrogram_data = processor.get_spectrogram_data()
        
        print(f"✅ Processing complete!")
        print(f"   Quantized bits: {len(quantized_bits) if quantized_bits else 0} bytes")
        print(f"   Spectrogram shape: {spectrogram_data.shape if spectrogram_data is not None else 'None'}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
    finally:
        processor.stop()
        processor.cleanup()
        print("🧹 Cleanup complete")
