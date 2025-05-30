import numpy as np
from datasets import Dataset, DatasetDict, DownloadMode
import datasets
import pandas as pd
import matplotlib.pyplot as plt
from DatasetGenerator import DatasetGenerator
import numpy as np

from scipy import signal

class DatasetHandler():
    def __init__(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.repo_name = repo_name
        self.dataFrame = None
        self.load_dataset()
        
    def load_dataset(self):
        path = self.repo_name+"/"+self.dataset_name
        print("Path: ", path)
        print("Config name: ", self.config_name)
        self.dataFrame = datasets.load_dataset(
            path,
            self.config_name,
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        )
        # self.get_dataframe_Info()
        if isinstance(self.dataFrame, DatasetDict):
            self.dataFrame = pd.concat([self.dataFrame[key].to_pandas() for key in self.dataFrame.keys()])
        # Check if dataframe is a dataset
        elif isinstance(self.dataFrame, Dataset):
            # convert to pandas dataframe
            self.dataFrame = self.dataFrame.to_pandas()
            
    def add_dataset(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        '''Add a new dataset to the existing dataset.'''
        new_dataset = datasets.load_dataset(repo_name+"/"+dataset_name, config_name)
        if isinstance(new_dataset, DatasetDict):
            new_dataset = pd.concat([new_dataset[key].to_pandas() for key in new_dataset.keys()])
        # Check if dataframe is a dataset
        elif isinstance(new_dataset, Dataset):
            # convert to pandas dataframe
            new_dataset = new_dataset.to_pandas()
        self.dataFrame = pd.concat([self.dataFrame, new_dataset], ignore_index=True)
        
    def get_dataframe_Info(self):
        print("DataFrame Info:")
        print(self.dataFrame.info())
        print("DataFrame Shape:", self.dataFrame.shape)
        print("DataFrame Columns:",self. dataFrame.columns)
        print("DataFrame Head:")
        print(self.dataFrame.head())
        print("DataFrame Description:")
        print(self.dataFrame.describe())
        
    def separate_iq_samples(self, data):
        '''Separate the IQ samples into I and Q components.'''
        # I is the second half of the samples
        I = data[len(data)//2:]
        # Q is the first half of the samples
        Q = data[:len(data)//2]
        return I, Q
    
    def convert_to_complex(self):
        '''Convert the loaded data to complex IQ samples.'''
        I = self.dataFrame["I"].to_numpy()
        Q = self.dataFrame["Q"].to_numpy()
        # data = self.dataFrame["data"]
        # I, Q = self.separate_iq_samples(data)
        data_complex = I + 1j*Q
        return data_complex
    
    def load_data(self):
        label = self.dataFrame["channel"]
        # label = self.dataFrame["label"]
        label = label.astype(int)
        label = np.transpose(label)
        data = self.convert_to_complex()
        return data,label
    
class ChannelSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=np.complex64)
        
        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
    
        return x_cropped

    def _gen_single_channel_spectrogram(self, sig, win_len=256, overlap=128):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.
        
        INPUT:
            SIG is the complex IQ samples.
            
            WIN_LEN is the window length used in STFT.
            
            OVERLAP is the overlap length used in STFT.
            
        RETURN:
            
            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig, 
                                window='boxcar', 
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)
        
        # Take the logarithm of the magnitude.      
        chan_spec_amp = np.log10(np.abs(spec)**2)
        return chan_spec_amp
    
    def channel_spectrogram(self, data, FFTwindow=512):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        print("Data shape:", data.shape)
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        print("Data shape:", data.shape)
        # Normalize the IQ samples.
        data = self._normalization(data)
            
        # Calculate the size of channel independent spectrograms.
        win_len=FFTwindow # 128 | 256 | 512 --Smaller window will give better time resolution but worse freq. resolution and vice versa. 128 for N=8192 is the most balanced
        overlap=win_len/2
        
        num_sample = data.shape[0]
        num_row = int(win_len*0.4)
        num_column = int(np.floor((data.shape[1]-win_len)/overlap + 1))
        data_channel_spec = np.zeros([num_sample, num_row, num_column, 1])
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_spec_amp = self._gen_single_channel_spectrogram(data[i],win_len, overlap)
            chan_spec_amp = self._spec_crop(chan_spec_amp)
            data_channel_spec[i,:,:,0] = chan_spec_amp
            
        return data_channel_spec

def plot_channel_spectrogram(data_channel_spec):
    '''Plot the channel independent spectrogram.'''
    plt.figure(figsize=(10, 5))
    plt.imshow(data_channel_spec[:,:,0], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.title('Channel Independent Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    
if __name__ == "__main__":
    # Example usage
    dataset_name = "Key-Generation"
    config_name = "Sinusoid-Powder-OTA-Dense-Nodes-123" #"Sinusoid-Powder-OTA-Lab"
    repo_name="CAAI-FAU"
    dataset_handler = DatasetHandler(dataset_name, config_name, repo_name)
    dataset_handler.get_dataframe_Info()
    data, labels = dataset_handler.load_data()
    signal_processor = ChannelSpectrogram()
    data_channel_spec = signal_processor.channel_spectrogram(data)
    plot_channel_spectrogram(data_channel_spec[0])
    plot_channel_spectrogram(data_channel_spec[1])
    plot_channel_spectrogram(data_channel_spec[2])
    plot_channel_spectrogram(data_channel_spec[3])
    