#This script is used to annotate the signal in the frequency domain using a moving window average.
#The script generates example signals and plots the FFT magnitude with detected start and end points.
#The annotate_signal function computes the FFT of the input signal and identifies the start and end points of the signal in the frequency domain.
#The plot_fft function plots the FFT magnitude with detected start and end points.
#The example signals include sinusoidal signal, chirp signal, modulated pulse, and random noise burst.
#The script also reads 11 modulation types from files and plots the FFT magnitude with detected start and end points for each modulation type.

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, gaussian
import os
import h5py

# Annotate Signal in Frequency Domain
def annotate_signal(signal, window_size=5, threshold_factor=0.01):
    """
    Annotates the start and end points of a signal in the frequency domain using a moving window average.

    Parameters:
        signal (np.ndarray): Complex-valued 1D signal (real + imaginary parts).
        window_size (int): Size of the window for computing average magnitude.
    Returns:
        (int, int): Start and end indices of the detected signal in the frequency domain.
    """
    # Compute FFT
    fft_signal = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_signal)
    fft_magnitude = np.fft.fftshift(fft_magnitude)
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    # Compute a search window for detecting the signal
    # Loop through the FFT magnitude signal with a moving window to take the average and detect the signal
    # when the average magnitude is above a certain threshold from the previous average magnitude
    start, end = 0, len(fft_magnitude)-1
    # Loop through the FFT magnitude signal with a moving window to take the average
    # and search for the start and ending points where the signal is present
    startFound, endFound = False, False
    
    for i in range(math.floor(len(fft_magnitude)/window_size)):
        starting_window = fft_magnitude[i*window_size:(i+1)*window_size]
        ending_window = fft_magnitude[len(fft_magnitude)-(i+1)*window_size:len(fft_magnitude)-i*window_size]
        avg_starting = np.mean(starting_window)
        avg_ending = np.mean(ending_window)
        if i == 0:
            prev_start_avg = avg_starting
            prev_end_avg = avg_ending
        else:
            if ((avg_starting - (prev_start_avg/(i+1))) > threshold_factor) and not startFound and start < end:
                start = i*window_size
                startFound = True
            if ((avg_ending - (prev_end_avg/(i+1))) > threshold_factor) and not endFound and end > start:
                endFound = True
                end = len(fft_magnitude)-(i+1)*window_size
            prev_start_avg += avg_starting
            prev_end_avg += avg_ending 
            if startFound and endFound:
                break
    start = start-window_size if start != 0 else start
    end = end+window_size if end != len(fft_magnitude)-1 else end
    return start, end, fft_magnitude

def plot_fft(signal, title="FFT Magnitude", window_size=5, threshold_factor=0.01):
    """
    Plots the FFT magnitude with detected start and end points.
    """
    start, end, fft_magnitude = annotate_signal(signal, window_size, threshold_factor)
    # Plot FFT Magnitude
    plt.figure(figsize=(10, 5))
    plt.plot(fft_magnitude, label="FFT Magnitude")
    # Mark detected start and end points
    if start is not None and end is not None:
        plt.scatter(start ,fft_magnitude[start], color="red", marker="x", label="Start Point")
        plt.scatter(end, fft_magnitude[end], color="green", marker="x", label="End Point")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

def plot_spec_with_annotations(filename):
    dataLabels = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "B-FM", "DSB-AM", "SSB-AM"]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown', 'black']
    with h5py.File(filename, 'r') as f:
        labels = f['labels'][:]
        data = f['data'][:]
        # data = data[:, 0, :] + 1j * data[:, 1, :]
        annotations = f['annotations'][:]
        # For each signal in the dataset calculate the fft magnitude as a row in the spectrogram
        spectrogram = []
        for i, signal in enumerate(data):
            fft_magnitude = np.abs(np.fft.fft(signal))
            fft_magnitude = np.fft.fftshift(fft_magnitude)
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
            spectrogram.append(fft_magnitude)       
        # Use the start and ending points as annotations for segmenting the modulation types in the spectrogram
        # Where the color depends of the modulation type of the signal for the start and end points
        # And the label is added only if the next label is different
        linewidth = 0.1
        alpha = 0.05
        plt.figure(figsize=(10, 5))
        plt.imshow(spectrogram, aspect='auto', cmap='viridis')
        for i, (start, end) in enumerate(annotations):
            if i == 0 or labels[i-1] != labels[i]:
                plt.text(start, i, dataLabels[labels[i]], color='black')
            plt.hlines(i, start, end, colors[labels[i]], linewidth=linewidth, alpha=alpha)
        plt.title("Spectrogram of FFT Magnitude with Detected Start and End Points")
        plt.xlabel("Frequency")
        plt.ylabel("Signal Index")
        plt.colorbar()
        plt.show()
    
window_size=10
threshold_factor=0.025
RXs = [3, 4, 5, 6] #[3, 4, 5, 6]

# Generate Example Signals
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, 1024, endpoint=False)  # Time vector

# # Example 1: Sinusoidal signal
# sin_signal = np.sin(2 * np.pi * 50 * t) + 1j * np.cos(2 * np.pi * 50 * t)
# plot_fft(sin_signal, title="FFT Magnitude - Sinusoidal Signal", window_size=window_size, threshold_factor=threshold_factor)

# # Example 2: Chirp signal (frequency sweep)
# chirp_signal = chirp(t, f0=10, f1=200, t1=1, method='linear') + 1j * chirp(t, f0=10, f1=200, t1=1, method='quadratic')
# plot_fft(chirp_signal, title="FFT Magnitude - Chirp Signal", window_size=window_size, threshold_factor=threshold_factor)

# # Example 3: Modulated pulse
# pulse = np.zeros(1024, dtype=complex)
# pulse[200:600] = np.exp(1j * 2 * np.pi * 100 * t[200:600]) * gaussian(400, std=50)
# plot_fft(pulse, title="FFT Magnitude - Modulated Pulse", window_size=window_size, threshold_factor=threshold_factor)

# # # Example 4: Random noise burst
# # noise = np.zeros(1024, dtype=complex)
# # noise[300:700] = (np.random.randn(400) + 1j * np.random.randn(400)) * 0.5
# # plot_fft(noise, title="FFT Magnitude - Random Noise Burst", window_size=window_size, threshold_factor=threshold_factor)

# Example 5: 11 Modulation types from files
# files = [f for f in os.listdir('.') if f.endswith('.dat')]
# print('Files: ' + str(files))
# for i, file in enumerate(files):
#     k = 0
#     # Read the .dat file as float32
#     data = np.fromfile(file, dtype=np.float32)
#     # Convert to complex IQ samples
#     complex_signal = data[::2] + 1j * data[1::2]
#     print(file)
#     plot_fft(complex_signal, title=file, window_size=window_size, threshold_factor=threshold_factor)


for rx in RXs:
    
    folder = '/Users/josea/Workspaces/siwn/data/nodes/'
    filename = folder+'Node_TX_1_RX_'+str(rx)+'_11Modulations_20250310.h5'
    new_filename = folder+'Node_TX_1_RX_'+str(rx)+'_11Modulations_20250310_annotated.h5'
    dataLabels = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "B-FM", "DSB-AM", "SSB-AM"]
    print("Plotting FFT Magnitude with Detected Start and End Points for RX: " + str(rx))
    with h5py.File(filename, 'r') as f:
        # Get single label index for every modulation type in the 'labels' dataset
        labels = f['labels'][:]
        # Get only one sample for every modulation type in the 'data' dataset
        data = f['data'][:]
        # Make the data complex by combining the real and imaginary parts
        data = data[:, 0, :] + 1j * data[:, 1, :]
        for i in range(11):
            complex_signal = data[np.where(labels == i)[0][0]]
            plot_fft(complex_signal, title="Modulation Type: " + str(dataLabels[i]) + ", RX:"+ str(rx), window_size=window_size, threshold_factor=threshold_factor)
    
    # print("Annotating Signals in Frequency Domain for RX: " + str(rx))
    # annotations = []
    # with h5py.File(filename, 'r') as f:
    #     labels = f['labels'][:]
    #     data = f['data'][:]
    #     data = data[:, 0, :] + 1j * data[:, 1, :]
    #     for signal in data:
    #         start, end, fft_magnitude = annotate_signal(signal, window_size, threshold_factor)
    #         annotations.append((start, end))
    # # Check if file exists
    # if os.path.exists(new_filename):
    #     os.remove(new_filename)
    # with h5py.File(new_filename, 'w') as f:
    #     f.create_dataset('data', data=data)
    #     f.create_dataset('labels', data=labels)
    #     f.create_dataset('annotations', data=annotations)
    
    # print("Plotting Spectrogram of FFT Magnitude with Detected Start and End Points for RX: " + str(rx))
    # plot_spec_with_annotations(new_filename)
    
