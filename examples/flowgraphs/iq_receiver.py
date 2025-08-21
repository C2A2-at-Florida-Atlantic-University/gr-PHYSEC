#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQReceiver GNU Radio Flowgraph
Receives and collects IQ samples using PlutoSDR.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'siwn-node', 'network'))

from gnuradio import gr, blocks
import threading
import time
import logging
import numpy as np

# Import GNU Radio IIO
try:
    from gnuradio import iio
    IIO_AVAILABLE = True
except ImportError as e:
    IIO_AVAILABLE = False
    print(f"Warning: GNU Radio IIO not available ({e})")

logger = logging.getLogger(__name__)

class IQReceiver(gr.top_block):
    """
    GNU Radio flowgraph for IQ sample collection
    """
    
    def __init__(self, sample_rate=1e6, center_freq=2.4e9, gain=10, 
                 vector_size=8192, sdr_uri="ip:192.168.2.1"):
        """
        Initialize IQ receiver flowgraph
        
        Args:
            sample_rate: Sample rate in Hz
            center_freq: SDR center frequency in Hz
            gain: RX gain
            vector_size: Number of samples to collect
            sdr_uri: PlutoSDR URI
        """
        gr.top_block.__init__(self, "IQ Receiver")
        
        # Parameters
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.vector_size = vector_size
        self.sdr_uri = sdr_uri
        self.is_running = False
        
        # Build the flowgraph
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Build the IQ receiver flowgraph"""
        
        # PlutoSDR source (if available)
        if IIO_AVAILABLE:
            try:
                self.pluto_source = iio.fmcomms2_source_fc32(self.sdr_uri, [True, True], 32768)
                self.pluto_source.set_frequency(int(self.center_freq))
                self.pluto_source.set_samplerate(int(self.sample_rate))
                self.pluto_source.set_gain_mode(0, 'slow_attack')
                self.pluto_source.set_gain(0, float(self.gain))
                self.source_available = True
                logger.info("Connected to PlutoSDR source")
            except Exception as e:
                logger.warning(f"Could not create PlutoSDR source: {e}")
                self.source_available = False
        else:
            self.source_available = False
            
        # Fallback: Test signal source
        if not self.source_available:
            from gnuradio import analog
            # Generate test signal with some noise
            self.test_source = analog.sig_source_c(
                self.sample_rate,
                analog.GR_SIN_WAVE,
                1000,  # 1 kHz tone
                0.5,   # Amplitude
                0      # Offset
            )
            
            self.noise_source = analog.noise_source_c(
                analog.GR_GAUSSIAN,
                0.1,   # Noise level
                0      # Seed
            )
            
            self.adder = blocks.add_cc(1)
            
        # Head block to limit number of samples
        self.head = blocks.head(gr.sizeof_gr_complex, self.vector_size)
        
        # Vector sink to collect samples
        self.vector_sink = blocks.vector_sink_c()
        
        # Optional: File sink for debugging
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex,
            "/tmp/iq_receiver_output.dat",
            False
        )
        
        # Throttle for test signal
        if not self.source_available:
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate, True)
        
        # Connections
        if self.source_available:
            self.connect((self.pluto_source, 0), (self.head, 0))
            logger.info("Connected to PlutoSDR source")
        else:
            # Test signal path
            self.connect((self.test_source, 0), (self.adder, 0))
            self.connect((self.noise_source, 0), (self.adder, 1))
            self.connect((self.adder, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.head, 0))
            logger.info("Using test signal source (PlutoSDR not available)")
            
        # Common connections
        self.connect((self.head, 0), (self.vector_sink, 0))
        self.connect((self.head, 0), (self.file_sink, 0))
        
    def start_reception(self):
        """Start IQ sample collection"""
        if not self.is_running:
            try:
                self.start()
                self.is_running = True
                logger.info(f"IQ reception started (collecting {self.vector_size} samples)")
            except Exception as e:
                logger.error(f"Failed to start reception: {e}")
                raise
                
    def stop_reception(self):
        """Stop the IQ sample collection and cleanup resources"""
        if self.is_running:
            try:
                self.stop()
                
                # Add timeout to prevent indefinite hanging
                import threading
                import time
                
                # Use a separate thread for wait() with timeout
                wait_thread = threading.Thread(target=self.wait)
                wait_thread.daemon = True
                wait_thread.start()
                wait_thread.join(timeout=5.0)  # 5 second timeout
                
                if wait_thread.is_alive():
                    logger.warning("GNU Radio receiver cleanup timed out")
                
            except Exception as e:
                logger.warning(f"Error during reception cleanup: {e}")
            finally:
                self.is_running = False
                
    def get_samples(self):
        """
        Get the collected IQ samples
        
        Returns:
            numpy.ndarray: Complex IQ samples or None if no data
        """
        try:
            data = self.vector_sink.data()
            if data:
                return np.array(data, dtype=np.complex64)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting samples: {e}")
            return None
            
    def reset_sink(self):
        """Reset the vector sink to clear old data"""
        if hasattr(self, 'vector_sink'):
            try:
                # Try different methods to clear the vector sink data
                if hasattr(self.vector_sink, 'reset'):
                    self.vector_sink.reset()
                elif hasattr(self.vector_sink, 'clear'):
                    self.vector_sink.clear()
                else:
                    # Recreate the vector sink as a last resort
                    from gnuradio import blocks
                    self.vector_sink = blocks.vector_sink_c()
                logger.debug("Vector sink reset successfully")
            except Exception as e:
                logger.warning(f"Could not reset vector sink: {e}")
                # Recreate the vector sink as fallback
                try:
                    from gnuradio import blocks
                    self.vector_sink = blocks.vector_sink_c()
                    logger.debug("Vector sink recreated")
                except:
                    pass
                
    def set_sample_rate(self, sample_rate):
        """Update the sample rate"""
        self.sample_rate = sample_rate
        if hasattr(self, 'throttle'):
            self.throttle.set_sample_rate(sample_rate)
        if self.source_available and hasattr(self, 'pluto_source'):
            self.pluto_source.set_samplerate(int(sample_rate))
            
    def set_center_frequency(self, center_freq):
        """Update the SDR center frequency"""
        self.center_freq = center_freq
        if self.source_available and hasattr(self, 'pluto_source'):
            self.pluto_source.set_frequency(int(center_freq))
            
    def set_gain(self, gain):
        """Update the RX gain"""
        self.gain = gain
        if self.source_available and hasattr(self, 'pluto_source'):
            self.pluto_source.set_gain(0, int(gain))


def create_iq_receiver(sample_rate=1e6, center_freq=2.4e9, gain=10,
                      vector_size=8192, sdr_uri="ip:192.168.2.1"):
    """
    Factory function to create an IQReceiver flowgraph
    
    Args:
        sample_rate: Sample rate in Hz
        center_freq: SDR center frequency in Hz
        gain: RX gain
        vector_size: Number of samples to collect
        sdr_uri: PlutoSDR URI
        
    Returns:
        IQReceiver: Configured flowgraph instance
    """
    return IQReceiver(sample_rate, center_freq, gain, vector_size, sdr_uri)


if __name__ == '__main__':
    # Test the IQReceiver flowgraph
    print("Testing IQReceiver flowgraph...")
    
    # Create receiver with test parameters
    receiver = create_iq_receiver(
        sample_rate=1e6,
        center_freq=2.4e9,
        gain=10,
        vector_size=1024  # Smaller for testing
    )
    
    try:
        print("Starting IQ reception...")
        receiver.start_reception()
        
        # Collect for 2 seconds
        time.sleep(2)
        
        # Get samples
        samples = receiver.get_samples()
        
        if samples is not None:
            print(f"✅ Reception test complete!")
            print(f"   Collected {len(samples)} samples")
            print(f"   Sample type: {samples.dtype}")
            print(f"   Sample range: {np.min(np.abs(samples)):.3f} to {np.max(np.abs(samples)):.3f}")
        else:
            print("❌ No samples collected")
        
    except Exception as e:
        print(f"❌ Reception test failed: {e}")
    finally:
        receiver.stop_reception()
        print("🧹 Cleanup complete")
