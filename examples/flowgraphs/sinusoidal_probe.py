#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SinusoidalProbe GNU Radio Flowgraph
Generates and transmits sinusoidal probe signals using PlutoSDR.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'siwn-node', 'network'))

from gnuradio import gr, analog, blocks
import threading
import time
import logging

# Import IIO utilities
try:
    import sys
    sys.path.append('/workspace/siwn/siwn-node/network')
    from iio_utils import create_modern_sink, set_sink_parameters
    IIO_AVAILABLE = True
except ImportError as e:
    IIO_AVAILABLE = False
    print(f"Warning: IIO utilities not available ({e})")

logger = logging.getLogger(__name__)

class SinusoidalProbe(gr.top_block):
    """
    GNU Radio flowgraph for sinusoidal probe transmission
    """
    
    def __init__(self, sample_rate=1e6, frequency=1e3, amplitude=0.5, 
                 center_freq=2.4e9, gain=10, sdr_uri="ip:192.168.2.1"):
        """
        Initialize sinusoidal probe flowgraph
        
        Args:
            sample_rate: Sample rate in Hz
            frequency: Sinusoid frequency in Hz  
            amplitude: Signal amplitude (0.0 to 1.0)
            center_freq: SDR center frequency in Hz
            gain: TX gain/attenuation
            sdr_uri: PlutoSDR URI
        """
        gr.top_block.__init__(self, "Sinusoidal Probe")
        
        # Parameters
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.amplitude = amplitude
        self.center_freq = center_freq
        self.gain = gain
        self.sdr_uri = sdr_uri
        self.is_running = False
        
        # Build the flowgraph
        self._build_flowgraph()
        
    def _build_flowgraph(self):
        """Build the sinusoidal probe flowgraph"""
        
        # Signal source: Sinusoid generator
        self.signal_source = analog.sig_source_c(
            self.sample_rate,      # Sample rate
            analog.GR_SIN_WAVE,    # Waveform type
            self.frequency,        # Frequency
            self.amplitude,        # Amplitude
            0                      # Offset
        )
        
        # Optional: Add noise for more realistic signal
        self.noise_source = analog.noise_source_c(
            analog.GR_GAUSSIAN, 
            0.01,  # Noise amplitude
            0      # Seed
        )
        
        self.adder = blocks.add_cc(1)
        
        # PlutoSDR sink (if available)
        if IIO_AVAILABLE:
            try:
                self.pluto_sink = create_modern_sink(self.sdr_uri, 32768)
                set_sink_parameters(self.pluto_sink, self.center_freq, self.sample_rate, self.gain)
                self.sink_available = True
            except Exception as e:
                logger.warning(f"Could not create PlutoSDR sink: {e}")
                self.sink_available = False
        else:
            self.sink_available = False
            
        # Fallback: File sink for testing
        if not self.sink_available:
            self.file_sink = blocks.file_sink(
                gr.sizeof_gr_complex,
                "/tmp/sinusoidal_probe_output.dat",
                False
            )
            
        # Throttle block to control rate when using file sink
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate, True)
        
        # Connections
        self.connect((self.signal_source, 0), (self.adder, 0))
        self.connect((self.noise_source, 0), (self.adder, 1))
        
        if self.sink_available:
            self.connect((self.adder, 0), (self.pluto_sink, 0))
            logger.info("Connected to PlutoSDR sink")
        else:
            self.connect((self.adder, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.file_sink, 0))
            logger.info("Using file sink (PlutoSDR not available)")
            
    def start_transmission(self):
        """Start the probe transmission"""
        if not self.is_running:
            try:
                self.start()
                self.is_running = True
                logger.info(f"Sinusoidal probe transmission started (freq={self.frequency}Hz)")
            except Exception as e:
                logger.error(f"Failed to start transmission: {e}")
                raise
                
    def stop_transmission(self):
        """Stop the probe transmission and cleanup resources"""
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
                    logger.warning("GNU Radio transmission cleanup timed out")
                
            except Exception as e:
                logger.warning(f"Error during transmission cleanup: {e}")
            finally:
                self.is_running = False
                
    def set_frequency(self, frequency):
        """Update the sinusoid frequency"""
        self.frequency = frequency
        if hasattr(self, 'signal_source'):
            self.signal_source.set_frequency(frequency)
            
    def set_amplitude(self, amplitude):
        """Update the signal amplitude"""
        self.amplitude = amplitude
        if hasattr(self, 'signal_source'):
            self.signal_source.set_amplitude(amplitude)
            
    def set_sample_rate(self, sample_rate):
        """Update the sample rate"""
        self.sample_rate = sample_rate
        if hasattr(self, 'signal_source'):
            self.signal_source.set_sampling_freq(sample_rate)
        if hasattr(self, 'throttle'):
            self.throttle.set_sample_rate(sample_rate)
        if self.sink_available and hasattr(self, 'pluto_sink'):
            self.pluto_sink.set_samplerate(int(sample_rate))
            
    def set_center_frequency(self, center_freq):
        """Update the SDR center frequency"""
        self.center_freq = center_freq
        if self.sink_available and hasattr(self, 'pluto_sink'):
            self.pluto_sink.set_frequency(int(center_freq))
            
    def set_gain(self, gain):
        """Update the TX gain"""
        self.gain = gain
        if self.sink_available and hasattr(self, 'pluto_sink'):
            self.pluto_sink.set_attenuation(0, float(gain))


def create_sinusoidal_probe(sample_rate=1e6, frequency=1e3, amplitude=0.5,
                           center_freq=2.4e9, gain=10, sdr_uri="ip:192.168.2.1"):
    """
    Factory function to create a SinusoidalProbe flowgraph
    
    Args:
        sample_rate: Sample rate in Hz
        frequency: Sinusoid frequency in Hz
        amplitude: Signal amplitude (0.0 to 1.0)
        center_freq: SDR center frequency in Hz
        gain: TX gain/attenuation
        sdr_uri: PlutoSDR URI
        
    Returns:
        SinusoidalProbe: Configured flowgraph instance
    """
    return SinusoidalProbe(sample_rate, frequency, amplitude, center_freq, gain, sdr_uri)


if __name__ == '__main__':
    # Test the SinusoidalProbe flowgraph
    print("Testing SinusoidalProbe flowgraph...")
    
    # Create probe with test parameters
    probe = create_sinusoidal_probe(
        sample_rate=1e6,
        frequency=1e3,      # 1 kHz tone
        amplitude=0.3,
        center_freq=2.4e9,
        gain=10
    )
    
    try:
        print("Starting probe transmission...")
        probe.start_transmission()
        
        # Transmit for 3 seconds
        time.sleep(3)
        
        print("✅ Transmission test complete!")
        
    except Exception as e:
        print(f"❌ Transmission test failed: {e}")
    finally:
        probe.stop_transmission()
        print("🧹 Cleanup complete")
