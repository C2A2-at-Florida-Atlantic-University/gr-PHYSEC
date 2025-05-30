import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Calculate the average power of the signal in DB given the real and imaginary samples
def calculate_power(real, imag):
    power = np.mean(real**2 + imag**2)
    power_db = 10*np.log10(power)
    return power_db

if __name__ == '__main__':
    print('Running Signal Analyzer')
    # Get all .dat files in the current directory
    files = [f for f in os.listdir('.') if f.endswith('.dat')]
    print('Files: ' + str(files))
    # Number of files
    numRows = len(files)
    plotTimeDomain = 1
    plotFrequencyDomain = 1
    plotSpectrogram = 1
    
    # Create subplots (3 rows per file: signal, spectrogram, and frequency domain)
    numColumns = plotTimeDomain+plotSpectrogram+plotFrequencyDomain
    fig, axes = plt.subplots(numRows, numColumns, figsize=(10*numColumns, 10*numRows))
    signalPowers = []
    # Loop through each file
    for i, file in enumerate(files):
        k = 0
        # Read the .dat file as float32
        data = np.fromfile(file, dtype=np.float32)
        # Convert to complex IQ samples
        complex_signal = data[::2] + 1j * data[1::2]
        
        power = calculate_power(np.real(complex_signal), np.imag(complex_signal))
        signalPowers.append(power)
            
        if plotTimeDomain:
            # Plot Time-domain Signal
            # Add both real and imaginary parts
            axes[i, k].plot(np.real(complex_signal), label="Real")
            axes[i, k].plot(np.imag(complex_signal), label="Imaginary")
            axes[i, k].set_title(f"Time-Domain Signal - {file}")
            axes[i, k].set_xlabel("Samples")
            axes[i, k].set_ylabel("Amplitude")
            axes[i, k].legend()
            k+=1
        
        if plotFrequencyDomain:
            # Plot Frequency-domain Signal
            # Calculate the FFT of the signal
            freq_signal = np.fft.fftshift(np.fft.fft(complex_signal))
            # Calculate the frequency axis
            freq_axis = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/1e6))  # Adjust d if needed
            # Plot the frequency domain signal
            axes[i, k].plot(freq_axis, 10 * np.log10(np.abs(freq_signal)))
            axes[i, k].set_title(f"Frequency-Domain Signal - {file}")
            axes[i, k].set_xlabel("Frequency [Hz]")
            axes[i, k].set_ylabel("Magnitude [dB]")
            k+=1
            
        if plotSpectrogram:
            f, t, Sxx = signal.spectrogram(complex_signal, fs=1e6, nperseg=256)  # Adjust fs if needed
            # Plot Spectrogram
            im = axes[i, k].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            axes[i, k].set_title(f"Spectrogram - {file}")
            axes[i, k].set_xlabel("Time [s]")
            axes[i, k].set_ylabel("Frequency [Hz]")
            fig.colorbar(im, ax=axes[i, k])
            k+=1
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    # Save plot as image
    plt.savefig('signal_spectrogram.png')
    
    # Plot signal powers in a scatter plot for every type of modulation and save it as an image
    plt.figure()
    plt.scatter(range(len(files)), signalPowers)
    plt.title('Power of the signals')
    plt.xlabel('Modulation')
    plt.ylabel('Power (dB)')
    # Add x labels as the file names
    plt.xticks(range(len(files)), files)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('power.png')
