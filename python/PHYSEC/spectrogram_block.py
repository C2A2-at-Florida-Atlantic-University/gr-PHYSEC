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
from scipy import signal

class spectrogram_block(gr.sync_block):
    """
    Channel spectrogram creation block.
    
    This block takes vector inputs of IQ samples and converts them to 
    channel-independent spectrograms using the same logic as dataset_preparation.py
    """
    
    def __init__(self, fft_window=512, vector_size=8192):
        gr.sync_block.__init__(
            self,
            name="PHYSEC Spectrogram Block",
            in_sig=[(np.complex64, vector_size)],  # Vector input
            out_sig=[(np.float32, (204, 31))]  # spectrogram output
        )
        
        # Store parameters
        self.fft_window = fft_window
        self.vector_size = vector_size
        
        print(f"PHYSEC Spectrogram Block initialized:")
        print(f"  FFT Window: {fft_window}")
        print(f"  Vector Size: {vector_size}")
        print(f"  Output Size: {204,31} (spectrogram)")
    
    def _normalization(self, data):
        """
        Normalize the signal by RMS, matching the dataset_preparation.py implementation.
        """
        try:
            s_norm = np.zeros(data.shape, dtype=np.complex64)
            
            for i in range(data.shape[0]):
                sig_amplitude = np.abs(data[i])
                rms = np.sqrt(np.mean(sig_amplitude**2))
                s_norm[i] = data[i]/rms
            
            return s_norm
        except Exception as e:
            print(f"Error in normalization: {e}")
            return data
    
    def _gen_single_channel_spectrogram(self, sig, win_len=256, overlap=128):
        """
        Generate single channel spectrogram matching dataset_preparation.py implementation.
        """
        try:
            # Short-time Fourier transform (STFT)
            f, t, spec = signal.stft(sig, 
                                    window='boxcar', 
                                    nperseg=win_len, 
                                    noverlap=overlap, 
                                    nfft=win_len,
                                    return_onesided=False, 
                                    padded=False, 
                                    boundary=None)
            
            # FFT shift to adjust the central frequency
            spec = np.fft.fftshift(spec, axes=0)
            
            # Take the logarithm of the magnitude
            chan_spec_amp = np.log10(np.abs(spec)**2)
            
            return chan_spec_amp
            
        except Exception as e:
            print(f"Error in _gen_single_channel_spectrogram: {e}")
            return None
    
    def _spec_crop(self, x):
        """
        Crop the generated channel independent spectrogram, matching dataset_preparation.py.
        """
        try:
            num_row = x.shape[0]
            x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
            return x_cropped
        except Exception as e:
            print(f"Error in spec_crop: {e}")
            return x
    
    def create_spectrogram(self, iq_data):
        """
        Create 2D spectrogram from IQ data matching the dataset_preparation.py workflow.
        
        Args:
            iq_data: Complex IQ samples of length vector_size
            
        Returns:
            2D spectrogram array or None if error
        """
        try:
            # Ensure we have the right number of samples
            if len(iq_data) != self.vector_size:
                print(f"Warning: Expected {self.vector_size} samples, got {len(iq_data)}")
                return None
            
            # Convert to complex array if needed
            if isinstance(iq_data, list):
                iq_data = np.array(iq_data)
            
            # Reshape to match the expected input format for the normalization function
            # The normalization function expects (num_samples, num_samples_per_packet)
            iq_data_reshaped = iq_data.reshape(1, -1)
            
            # Use the exact same parameters as in dataset_preparation.py
            win_len = self.fft_window
            overlap = win_len/2
            
            print(f"Using FFT parameters: FFTwindow={self.fft_window}, win_len={win_len}, overlap={overlap}")
            
            # Normalize the IQ samples (matching dataset_preparation.py)
            iq_data_normalized = self._normalization(iq_data_reshaped)
            
            # Calculate the size of channel independent spectrograms
            num_row = int(win_len*0.4)  # 40% of frequency bins
            num_column = int(np.floor((iq_data_normalized.shape[1]-win_len)/overlap + 1))
            
            print(f"Expected spectrogram dimensions: num_row={num_row}, num_column={num_column}")
            
            # Generate spectrogram using the same method as dataset_preparation.py
            chan_spec_amp = self._gen_single_channel_spectrogram(iq_data_normalized[0], win_len, overlap)
            
            if chan_spec_amp is None:
                return None
            
            # Apply the same cropping as in dataset_preparation.py
            chan_spec_amp = self._spec_crop(chan_spec_amp)
            
            print(f"Raw spectrogram shape after processing: {chan_spec_amp.shape}")
            
            spectrogram_final = chan_spec_amp
            print(f"Final reshaped spectrogram: {spectrogram_final.shape}")
            return spectrogram_final
            
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
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
            # Get input data (vector of complex samples)
            in0 = input_items[0]
            out0 = output_items[0]
            num_input_items = len(in0)
            
            print(f"Processing {num_input_items} vector(s) of size {self.vector_size}")
            
            for i in range(num_input_items):
                # Get the current vector of samples
                iq_vector = in0[i]
                
                # Create 2D spectrogram
                spectrogram = self.create_spectrogram(iq_vector)
                print(f"Spectrogram shape: {spectrogram.shape}")
                # return the 2D spectrogram
                out0[i] = spectrogram
            
            return num_input_items
            
        except Exception as e:
            print(f"Error in work method: {e}")
            return 0


if __name__ == "__main__":
    # Test the spectrogram block
    print("Testing spectrogram block...")
    
    # Create test data
    test_iq = np.random.randn(8192) + 1j * np.random.randn(8192)
    test_iq = test_iq.astype(np.complex64)
    
    # Create block instance
    block = spectrogram_block(fft_window=512, vector_size=8192)
    
    # Test spectrogram creation
    spectrogram = block.create_spectrogram(test_iq)
    if spectrogram is not None:
        print(f"✓ Test successful! Spectrogram shape: {spectrogram.shape}")
    else:
        print("✗ Test failed!")
