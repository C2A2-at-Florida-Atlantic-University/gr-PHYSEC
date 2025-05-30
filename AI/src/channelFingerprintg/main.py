from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from dataset_preparation import ChannelSpectrogram, LoadDatasetChannels
from deep_learning_models import identity_loss, QuadrupletNet_Channel

import time

#%%

def train_channel_feature_extractor(
        #file_path = './dataset/Train/dataset_training_aug.h5', 
        file_path = './LoRa_RFFI-main/dataset/Train/IQ_Samples_SDR_12345.h5',
        folder_models = './models/',
        epochs=1000
                            ):
    '''
    train_feature_extractor trains an RFF extractor using triplet loss.
    
    INPUT: 
        FILE_PATH is the path of training dataset.
        
        DEV_RANGE is the label range of LoRa devices to train the RFF extractor.
        
        PKT_RANGE is the range of packets from each device to train the RFF extractor.
        
        SNR_RANGE is the SNR range used in data augmentation. 
        
    RETURN:
        FEATURE_EXTRACTOR is the RFF extractor which can extract features from
        channel-independent spectrograms.
    '''
    
    LoadDatasetObj = LoadDatasetChannels()
    
    # Load preamble IQ samples and labels.
    data,label = LoadDatasetObj.load_iq_samples(file_path)
    
    # Add additive Gaussian noise to the IQ samples.
    #data = awgn(data, snr_range)
    
    ChannelSpectrogramObj = ChannelSpectrogram()
    
    # Convert time-domain IQ samples to channel-independent spectrograms.
    fft_len = 512
    data = ChannelSpectrogramObj.channel_spectrogram(data,fft_len)

    print("len: ", len(data))
    print("shape: ", data.shape)

    data_length = len(data)
    alpha = 0.5
    beta = 0.5
    gamma = 0.2

    batch_size = 40
    patience = 20

    #NetObj =  TripletNet_Channel()
    NetObj = QuadrupletNet_Channel()
    
    # Create an RFF extractor.
    feature_extractor = NetObj.feature_extractor(data.shape)

    feature_extractor.summary()
    
    # Create the quadruplet net using the RFF extractor.
    #net = NetObj.create_triplet_net(feature_extractor, alpha1)
    net = NetObj.create_quadruplet_net(feature_extractor, alpha, beta, gamma)

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    early_stop = EarlyStopping('val_loss', 
                            min_delta = 0, 
                            patience = 
                            patience)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                min_delta = 0, 
                                factor = 0.2, 
                                patience = 10, 
                                verbose=1
                                )
    callbacks = [early_stop, reduce_lr]
    validation_size=0.1
    # Split the dasetset into validation and training sets.
    data_train, data_valid, label_train, label_valid = train_test_split(data, 
                                                                        label, 
                                                                        test_size=validation_size, 
                                                                        shuffle= False)
    del data, label

    print(data_train.shape) 
    print(data_valid.shape)

    print(label_train.shape)
    print(label_valid.shape)
    
    # Create the trainining generator.
    train_generator = NetObj.create_generator_channel(batch_size, 
                                                     data_train, 
                                                     label_train)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                     data_valid, 
                                                     label_valid)
    
    
    # Use the RMSprop optimizer for training.
    LearningRate = 1e-3
    opt = RMSprop(learning_rate=LearningRate)

    net.compile(
        loss = identity_loss,
        #metrics=['accuracy'],
        optimizer = opt)

    print("steps_per_epoch", data_train.shape[0]//batch_size)
    print("validation_steps" , data_valid.shape[0]//batch_size)
    # Start training.
    history = net.fit(train_generator,
                            steps_per_epoch = data_train.shape[0]//batch_size,
                            epochs = epochs,
                            validation_data = valid_generator,
                            validation_steps = data_valid.shape[0]//batch_size,
                            verbose=1, 
                            callbacks = callbacks)

    print(history.history.keys())

    timestamp = time.time()

    feature_extractor.save(folder_models+'QExtractor2_'+str(fft_len)+'_alpha'+str(alpha)+'_batch'+str(batch_size)+'_val'+str(validation_size)+'_RMS'+str(LearningRate)+'_DSsin2.4dev1278-'+str(data_length)+'_'+str(timestamp)+'.h5')
    return feature_extractor

if __name__ == '__main__':
    

    run_for = 'Train Channel Fingerprinting'

    if run_for == 'Train Channel Fingerprinting':

        #folder = '/home/siwn/AI/'
        folder = "/Users/josea/Research/Models/"
        #file_path = folder+'Dataset_Channels_sinusoid_dev_1287_freq_2.4e9_sr_1e6_gain_0_60_4800_S.hdf5'
        file_path = "Users/josea/Research/siwn-node/orch/Dataset_Channels_1710226687.hdf5"
        # Train an RFF extractor.
        # Save the trained model.
        feature_extractor = train_channel_feature_extractor(file_path = file_path, folder_models = folder, epochs=100)
        

# %%
