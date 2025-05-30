from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import RMSprop
from deep_learning_models import identity_loss, QuadrupletNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram

import time

def train_channel_feature_extractor(dataset, epochs=1000):
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
    
    # Load preamble IQ samples and labels.
    data,label = dataset
    
    # Add additive Gaussian noise to the IQ samples.
    #data = awgn(data, snr_range)
    
    ChannelSpectrogramObj = ChannelSpectrogram()
    
    # Convert time-domain IQ samples to channel-independent spectrograms.
    fft_len = 256
    data = ChannelSpectrogramObj.channel_spectrogram(data,fft_len)

    print("len: ", len(data))
    print("shape: ", data.shape)

    data_length = len(data)
    alpha = 0.5
    beta = 0.5
    gamma = 0.2

    batch_size = 64
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
                                    factor = 0.1, 
                                    patience = 20, 
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
    
    # Create the trainining generator.
    train_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_train, 
                                                        label_train)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_valid, 
                                                        label_valid)
    
    # Use the RMSprop optimizer for training.
    LearningRate = 0.1
    opt = RMSprop(learning_rate=LearningRate)

    net.compile(
        loss = identity_loss,
        #metrics=['accuracy'],
        optimizer = opt)

    print("Training data:", data_train.shape)
    print("Validation data:", data_valid.shape)
    # Start training.
    history = net.fit(train_generator,
                        steps_per_epoch = data_train.shape[0]//batch_size,
                        epochs = epochs,
                        validation_data = valid_generator,
                        validation_steps = data_valid.shape[0]//batch_size,
                        verbose=1, 
                        callbacks = callbacks)

    timestamp = time.time()
    
    savedFile = 'QExtractor2_'+str(fft_len)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_batch'+str(batch_size)+'_val'+str(validation_size)+'_RMS'+str(LearningRate)+'_DSsin2.4dev1278-'+str(data_length)+'_'+str(timestamp)+'.h5'

    feature_extractor.save(savedFile)
    
    print("Saving file: ", savedFile)
    
    return feature_extractor

if __name__ == "__main__":
    node_configurations = {
        'OTA-lab': {
            'dataset_name': 'Key-Generation',
            'config_name': 'Sinusoid-Powder-OTA-Lab-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': [
                [1,2,3],
                [1,4,5],
                [1,4,8],
                [2,4,3],
                [4,2,5],
                [4,2,8],
                # [4,8,5],
                # [5,7,8],
                # [5,8,7],
                # [8,4,1],
                # [8,5,1],
                # [8,5,4]
            ]
        },
        'OTA-Dense': {
            'dataset_name': 'Key-Generation',
            'config_name': 'Sinusoid-Powder-OTA-Dense-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': [
                # [1,2,3],
                [1,2,5],
                # [1,3,2],
                # [4,3,5]
            ]
        }
    }
    configuration = node_configurations['OTA-lab']
    
    dataset_name = configuration['dataset_name']
    repo_name = configuration['repo_name']
    node_Ids = configuration['node_Ids']
    # Load the dataset first and feed that to the model
    for idx, node_ids in enumerate(node_Ids):
        config_name = configuration['config_name']+"-"+"".join(str(node) for node in node_ids)
        print("Config name: ", config_name)
        if idx == 0:
            dataset = DatasetHandler(dataset_name, config_name, repo_name)
        else:
            dataset.add_dataset(dataset_name, config_name, repo_name)
    dataset.get_dataframe_Info()
    # Train an RFF extractor.
    # Save the trained model.
    feature_extractor = train_channel_feature_extractor(dataset.load_data(), epochs=1000)