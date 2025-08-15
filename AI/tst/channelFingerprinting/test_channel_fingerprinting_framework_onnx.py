import tensorflow as tf
from statistics import mean
import numpy as np
import h5py
import unireedsolomon as rs
from tqdm import tqdm
import matplotlib.pyplot as plt
from nistrng import *
import hashlib
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sys

# Add dataset preparation path at gr-PHYSEC/AI/dataset_preparation.py
sys.path.append('/workspace/gr-PHYSEC/AI')
from dataset_preparation import awgn, ChannelSpectrogram, LoadDatasetChannels

# Need to add ONNX support to the feature extractor model
# from onnx import load
import onnxruntime as ort

def feature_quantization(features):
    mean_features = mean(features)
    threshold = mean_features #0
    features_quatized = []
    for i in features:
        if i >= threshold:
            features_quatized.append(1)
        else:
            features_quatized.append(0)
    return features_quatized  

def arr2str(arr):
    str_arr = ''
    for i in arr:
        str_arr += str(i)
    return str_arr

def str2arr(string):
    arr = []
    integer = int(string, 16)
    binary_string = format(integer, '0>42b')
    for i in binary_string:
        arr.append(int(i))
    return arr

def reconcile(A,B,n=255,k=128):
    A = hex(int(arr2str(A), 2))
    A = str(A[2:]) #A hex key
    B = hex(int(arr2str(B), 2))
    B = str(B[2:]) #B hex key
    coder = rs.RSCoder(n,k)
    Aencode = coder.encode(A) #Encode A
    Aparity = Aencode[k:] #A parity bits
    BAparity = B+Aparity #B key + A parity bits
    try:
        Bdecode = coder.decode(BAparity) #Decode B key + A parity bits
        Breconciled = Bdecode[0] #Reconcilieted key
    except:
        Breconciled = B
    return [A == Breconciled,Breconciled]

def reconciliation_rate(data,n,k):
    j = 0
    reconciliation_data1 = []
    reconciliation_data2 = []
    reconciliation_data3 = []
    reconciled_data = []
    pbar = tqdm(total = data.shape[0]/4+1)
    
    while j <= data.shape[0]-3:
        if j == 0:
            print("Key length: ", len(data[j]))
            print("Key: ", data[j])
            print("N: ", n)
            print("K: ", k)
            hex_key = hex(int(arr2str(data[j]), 2))
            print("Hex key: ", hex_key)
            hex_key = str(hex_key[2:])
            print("String key: ", hex_key)
            coder = rs.RSCoder(n,k)
            hex_encoded = coder.encode(hex_key)
            print("Encoded key: ", hex_encoded)
            hex_parity = hex_encoded[k:]
            print("Parity bits: ", hex_parity)
            # return
        reconciliation_data1.append(reconcile(data[j],data[j+2],n,k))
        reconciliation_data2.append(reconcile(data[j],data[j+1],n,k))
        reconciliation_data3.append(reconcile(data[j+2],data[j+3],n,k))
        j = j + 4
        pbar.update(1)
        
    return reconciliation_data1, reconciliation_data2, reconciliation_data3

def privacyAmplification(data):
    # encode the string
    encoded_str = data.encode()
    # create sha3-256 hash objects
    obj_sha3_256 = hashlib.new("sha3_512", encoded_str)
    return(obj_sha3_256.hexdigest())

#https://github.com/InsaneMonster/NistRng/blob/master/benchmarks/numpy_rng_test.py
def NIST_RNG_test(data):
    #Eligible test from NIST-SP800-22r1a:
    #-monobit
    #-runs
    #-dft
    #-non_overlapping_template_matching
    #-approximate_entropy
    #-cumulative sums
    #-random_excursion
    #-random_excursion_variant
    eligible_battery: dict = check_eligibility_all_battery(np.array(data[0]), SP800_22R1A_BATTERY)
    num_packets = len(data)
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
    results_passed = {"Monobit" : [],"Frequency Within Block" : [],"Runs" : [],"Longest Run Ones In A Block" : [],
                      "Discrete Fourier Transform" : [],"Non Overlapping Template Matching" : [],"Serial" : [],
                      "Approximate Entropy" : [],"Cumulative Sums" : [],"Random Excursion" : [],
                      "Random Excursion Variant" : []}
    results_score = {"Monobit" : [],"Frequency Within Block" : [],"Runs" : [],"Longest Run Ones In A Block" : [],
                     "Discrete Fourier Transform" : [],"Non Overlapping Template Matching" : [],"Serial" : [],
                     "Approximate Entropy" : [],"Cumulative Sums" : [],"Random Excursion" : [],
                     "Random Excursion Variant" : []}
    data_results = []
    for i in range(0,len(data)):
        eligible_battery: dict = check_eligibility_all_battery(np.array(data[i]), SP800_22R1A_BATTERY)
        results = run_all_battery(np.array(data[i]), eligible_battery, False)
        data_results.append(results)
        for result, elapsed_time in results:
            score = np.round(result.score, 3)
            name = result.name
            results_score[name].append(score)
            if result.passed:
                passed = 1
            else:
                passed = 0
            results_passed[name].append(passed)
            
    for i in results_score:
        passing_score = sum(results_passed[i])/num_packets
        if round(passing_score) == 1:
            print("- PASSED ("+str(passing_score)+") - score: " + str(np.round(sum(results_score[i])/num_packets, 3)) + " - " + i)
        else:
            print("- FAILED ("+str(passing_score)+") - score: " + str(np.round(sum(results_score[i])/num_packets, 3)) + " - " + i)

    ith = 0
    plt.figure(figsize=[20,20])
    for i in results_passed:
        #print(results_passed[i])
        ith = ith + 1
        plt.subplot(4, 3,ith)
        plt.plot(results_passed[i])
        # naming the x axis
        plt.xlabel('N Key Generated')
        # naming the y axis
        plt.ylabel('Passed')
        plt.ylim(-0.05, 1.05)
        plt.title(i)
        
def KDR(A,B):
    kdr = np.bitwise_xor(A,B)
    kdr = np.sum(kdr)
    kdr = kdr/len(A)
    return kdr
    
def KDR_data(data):
    j = 0
    KDR_AB = []
    KDR_AC = []
    KDR_BC = []
    pbar = tqdm(total = data.shape[0]/4+1)
    while j <= data.shape[0]-3:
        KDR_AB.append(KDR(data[j],data[j+2]))
        KDR_AC.append(KDR(data[j],data[j+1]))
        KDR_BC.append(KDR(data[j+2],data[j+3]))
        j = j + 4
        pbar.update(1)
    return KDR_AB, KDR_AC, KDR_BC

PHYSEC_dir = '/workspace/gr-PHYSEC/'

model_path = PHYSEC_dir+'models/'

feature_extractor_name = model_path+'QExtractor.h5'
feature_extractor_onnx_name = model_path+'QExtractor.onnx'

#TESTING DATASETS
dataset_path = PHYSEC_dir+'datasets/'

dataset = dataset_path+'Dataset_Channels_sinusoid_dev_871_1690302750.hdf5'

#with h5py.File(dataset, 'r') as f:
#    data = np.array(f['data'][:])

#data = np.array(data)  
#print(data[0])

try:
    with tf.device('/CPU:0'):
        print("loading model")
        feature_extractor = load_model(feature_extractor_name)
except: # Load onnx model instead
    print("loading onnx model")
    # feature_extractor = ort.InferenceSession(feature_extractor_onnx_name)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(
        feature_extractor_onnx_name, 
        providers=providers
    )
    
    # Get model input/output information
    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]
    
#feature_extractor.summary()
#plot_model(feature_extractor, to_file='featureExtractor.png', show_shapes=True, show_layer_names=True)

testLevel = 4
enable_flags = {
    'test_post_processing': testLevel >= 0,
    'test_feature_extractor': testLevel >= 1,
    'test_feature_quantization': testLevel >= 2,
    'test_kdr': testLevel >= 3,
    'test_reconciliation': testLevel >= 4,
    'test_privacy_amplification': testLevel >= 5,
    'test_nist_rng': testLevel >= 6,
    'plot_data': False
}

save_results = True
compare_results = True
test_fingerprint_block_results = True

if enable_flags['test_post_processing']:
    #with tf.device('/GPU:0'):
    LoadDatasetObj = LoadDatasetChannels()
    ChannelSpectrogramObj = ChannelSpectrogram()
    # Load the classification dataset. (IQ samples and labels)
    print("loading data")
    data, labels = LoadDatasetObj.load_iq_samples(dataset)
    print(data.shape)
    #data = data[0:1]
    raw_data = data

    print(data[0])
    # Convert IQ samples to channel independent spectrograms. (classification data)
    t_start = time.time()
    print("converting IQ to spectrograms")
    data = ChannelSpectrogramObj.channel_spectrogram(np.array(data),512)
    t_end = time.time()
    print("Time for signal processing: ", t_end-t_start)

    print(data.shape)
    
    if enable_flags['test_feature_extractor']:
        t_start = time.time()
        # Extract RFFs from channel independent spectrograms.
        print("extracting features")
        # features = feature_extractor.predict(data)
        # features = ort_session.run(None, {'input': data})
        input_name = ort_session.get_inputs()[0].name
        # Run inference
        features = ort_session.run(
            None, 
            {input_name: data.astype(np.float32)}
        )[0]
        print(features.shape)
            
        t_end = time.time()
        print("Time for fingerprinting: ", t_end-t_start)
        #print(features)
        del data
        #print(features[0])

        # Separate real and imaginary parts
        real_parts = np.real(raw_data)
        imaginary_parts = np.imag(raw_data)
        print(real_parts[0])
        print(imaginary_parts[0])
        if enable_flags['plot_data']:
            plt.plot(real_parts[0], color='red')
            plt.plot(imaginary_parts[0], color='blue')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.title('Complex Data Plot')
            plt.show()

        if enable_flags['test_feature_quantization']:
            #feature quantization
            quantized_data = []
            t_start = time.time()
            for i in features:
                features_quatized = feature_quantization(i)
                quantized_data.append(features_quatized)

            quantized_data = np.array(quantized_data)
            t_end = time.time()
            print("Time for quantization: ", t_end-t_start)
            print(len(quantized_data[0]))
            print(len(quantized_data))
            
            if save_results:
                # Save quantized data in a hdf5 file
                with h5py.File(dataset_path+'Dataset_Channels_quantized_onnx_dev_871_1690302750.hdf5', 'w') as f:
                    f.create_dataset('quantized_keys', data=quantized_data)
                print("Saved quantized data to hdf5 file")
            if compare_results:
                # Load quantized data from hdf5 file
                with h5py.File(dataset_path+'Dataset_Channels_quantized_TF38Model_dev_871_1690302750.hdf5', 'r') as f:
                    quantized_data_TF38 = np.array(f['quantized_keys'][:])                    
                    # Compare quantized data
                    if np.array_equal(quantized_data, quantized_data_TF38):
                        print("Quantized data matches TF38Model")
                    else:
                        print("Quantized data does not match TF38Model")
                        
                
            if test_fingerprint_block_results:
                # Load quantized data from hdf5 file
                with h5py.File(dataset_path+'Dataset_Channels_quantized_fingerprint_block_work_test.hdf5', 'r') as f:
                    quantized_data_fingerprint_block = np.array(f['quantized_features'][:])
                    # Compare quantized data
                    if np.array_equal(quantized_data, quantized_data_fingerprint_block):
                        print("Quantized data matches fingerprint block")
                    else:
                        print("Quantized data does not match fingerprint block")
                        
                with h5py.File(dataset_path+'Dataset_Channels_quantized_feature_quantization_block_work_test.hdf5', 'r') as f:
                    quantized_data_feature_quantization_block = np.array(f['quantized_features'][:])
                    # Compare quantized data
                    if np.array_equal(quantized_data, quantized_data_feature_quantization_block):
                        print("Quantized data matches feature quantization block")
                    else:
                        print("Quantized data does not match feature quantization block")
            
            if enable_flags['test_kdr']:

                def groupAverage(arr, n):
                    result = []
                    i=0
                    while i <len(arr):
                        sum_n = 0;
                        j = 0
                        while j < n:
                            sum_n = sum_n + arr[i]
                            j = j + 1
                        result.append(sum_n/n);
                        i = i + n
                    return result

                quantized_data = quantized_data[0:400]

                KDR_AB, KDR_AC, KDR_BC = KDR_data(quantized_data)
                KDR_AB_average = np.sum(KDR_AB)/(len(KDR_AB))
                print("Average KDR Alice-Bob:", KDR_AB_average)
                KDR_AC_average = np.sum(KDR_AC)/(len(KDR_AC))
                print("Average KDR Alice-Eve:", KDR_AC_average)
                KDR_BC_average = np.sum(KDR_BC)/(len(KDR_BC))
                print("Average KDR Bob-Eve:", KDR_BC_average)

                batch_size = 2
                KDR_AB_average_batch = groupAverage(KDR_AB, batch_size)
                KDR_AC_average_batch = groupAverage(KDR_AC, batch_size)
                KDR_BC_average_batch = groupAverage(KDR_BC, batch_size)


                plt.figure(figsize=[15,3])

                plt.subplot(1, 3,1)
                plt.plot(KDR_AB)
                # naming the x axis
                plt.xlabel('N Key Generated')
                # naming the y axis
                plt.ylabel('KDR')
                plt.ylim(-0.05, 1)
                plt.title("KDR Alice Bob")

                plt.subplot(1, 3,2)
                plt.plot(KDR_AC)
                # naming the x axis
                plt.xlabel('N Key Generated')
                # naming the y axis
                plt.ylabel('KDR')
                plt.ylim(-0.05, 1)
                plt.title("KDR Alice Eve")

                plt.subplot(1, 3,3)
                plt.plot(KDR_BC)
                # naming the x axis
                plt.xlabel('N Key Generated')
                # naming the y axis
                plt.ylabel('KDR')
                plt.ylim(-0.05, 1)
                plt.title("KDR Bob Eve")

                # KDR Bar
                barWidth = 0.25
                fig = plt.subplots(figsize =(12, 8))
                
                # set height of bar
                AB = KDR_AB_average_batch
                AE = KDR_AC_average_batch
                BE = KDR_BC_average_batch
                
                # Set position of bar on X axis
                br1 = np.arange(len(AB))
                br2 = [x + barWidth for x in br1]
                br3 = [x + barWidth for x in br2]
                
                # Make the plot
                plt.bar(br1, AB, color ='b', width = barWidth,
                        edgecolor ='grey', label ='Alice-Bob')
                plt.bar(br2, AE, color ='r', width = barWidth,
                        edgecolor ='grey', label ='Alice-Eve')
                plt.bar(br3, BE, color ='m', width = barWidth,
                        edgecolor ='grey', label ='Bob-Eve')
                
                # Adding Xticks
                plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
                plt.ylabel('BDR', fontweight ='bold', fontsize = 15)
                plt.xticks([r + barWidth for r in range(len(AB))],
                    np.arange(1, len(AB)+1)*batch_size)

                plt.title("Bit Dissagreement Ratio Scenario 1")
                plt.legend()
                plt.show()

                if enable_flags['test_reconciliation']:
                    #Reconciliation
                    k = int(len(quantized_data[0])/4)

                    n1 = int(k+(k/1)-1)
                    print(n1)
                    s1 = int(n1-k)
                    t_start = time.time()
                    reconciliation11,reconciliation21,reconciliation31=reconciliation_rate(quantized_data,n1,k)
                    t_end = time.time()
                    print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
                    #print(reconciliation11)

                    n2 = int(k+(k/2)-1)
                    print(n2)
                    s2 = int(n2-k)
                    t_start = time.time()
                    reconciliation12,reconciliation22,reconciliation32=reconciliation_rate(quantized_data,n2,k)
                    t_end = time.time()
                    print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
                    #print(reconciliation12)

                    n3 = int(k+(k/3)-1)
                    print(n3)
                    s3 = int(n3-k)
                    t_start = time.time()
                    reconciliation13,reconciliation23,reconciliation33=reconciliation_rate(quantized_data,n3,k)
                    t_end = time.time()
                    print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
                    #print(reconciliation13)


                    rec_rate_data11 = []
                    rec_rate_data21 = []
                    rec_rate_data31 = []

                    rec_rate_data12 = []
                    rec_rate_data22 = []
                    rec_rate_data32 = []

                    rec_rate_data13 = []
                    rec_rate_data23 = []
                    rec_rate_data33 = []

                    i = 0
                    while i < len(reconciliation11):
                        rec_rate_data11.append(reconciliation11[i][0])
                        rec_rate_data21.append(reconciliation21[i][0])
                        rec_rate_data31.append(reconciliation31[i][0])
                        
                        rec_rate_data12.append(reconciliation12[i][0])
                        rec_rate_data22.append(reconciliation22[i][0])
                        rec_rate_data32.append(reconciliation32[i][0])
                        
                        rec_rate_data13.append(reconciliation13[i][0])
                        rec_rate_data23.append(reconciliation23[i][0])
                        rec_rate_data33.append(reconciliation33[i][0])
                        i = i+1

                    rec_rate_AB_average1 = np.sum(rec_rate_data11)/(len(rec_rate_data11))
                    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average1)
                    rec_rate_AC_average1 = np.sum(rec_rate_data21)/(len(rec_rate_data21))
                    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average1)
                    rec_rate_BC_average1 = np.sum(rec_rate_data31)/(len(rec_rate_data31))
                    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average1)

                    rec_rate_AB_average2 = np.sum(rec_rate_data12)/(len(rec_rate_data12))
                    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average2)
                    rec_rate_AC_average2 = np.sum(rec_rate_data22)/(len(rec_rate_data22))
                    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average2)
                    rec_rate_BC_average2 = np.sum(rec_rate_data32)/(len(rec_rate_data32))
                    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average2)

                    rec_rate_AB_average3 = np.sum(rec_rate_data13)/(len(rec_rate_data13))
                    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average3)
                    rec_rate_AC_average3 = np.sum(rec_rate_data23)/(len(rec_rate_data23))
                    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average3)
                    rec_rate_BC_average3 = np.sum(rec_rate_data33)/(len(rec_rate_data33))
                    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average3)

                    plt.figure(figsize=[15,3])
                    plt.subplot(1, 3,1)
                    plt.plot(rec_rate_data11)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Bob for RS("+str(n1)+","+str(k)+")")

                    plt.subplot(1, 3,2)
                    plt.plot(rec_rate_data21)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Eve for RS("+str(n1)+","+str(k)+")")

                    plt.subplot(1, 3,3)
                    plt.plot(rec_rate_data31)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Bob Eve for RS("+str(n1)+","+str(k)+")")

                    plt.figure(figsize=[15,3])
                    plt.subplot(1, 3,1)
                    plt.plot(rec_rate_data12)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Bob for RS("+str(n2)+","+str(k)+")")

                    plt.subplot(1, 3,2)
                    plt.plot(rec_rate_data22)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Eve for RS("+str(n2)+","+str(k)+")")

                    plt.subplot(1, 3,3)
                    plt.plot(rec_rate_data32)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Bob Eve for RS("+str(n2)+","+str(k)+")")

                    plt.figure(figsize=[15,3])
                    plt.subplot(1, 3,1)
                    plt.plot(rec_rate_data13)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Bob for RS("+str(n3)+","+str(k)+")")

                    plt.subplot(1, 3,2)
                    plt.plot(rec_rate_data23)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Alice Eve for RS("+str(n3)+","+str(k)+")")

                    plt.subplot(1, 3,3)
                    plt.plot(rec_rate_data33)
                    # naming the x axis
                    plt.xlabel('N Batch')
                    # naming the y axis
                    plt.ylabel('Reconciliation Rate')
                    plt.ylim(-0.05, 1.05)
                    plt.title("Reconciliation Rate Bob Eve for RS("+str(n3)+","+str(k)+")")

                    len(reconciliation11)
                    print(reconciliation11[2][0])

                    # set width of bar
                    barWidth = 0.25
                    fig = plt.subplots(figsize =(12, 8))
                    
                    # set height of bar
                    AB = [rec_rate_AB_average1, rec_rate_AB_average2, rec_rate_AB_average3]
                    AE = [rec_rate_AC_average1, rec_rate_AC_average2, rec_rate_AC_average3]
                    BE = [rec_rate_BC_average1, rec_rate_BC_average2, rec_rate_BC_average3]

                    # Set position of bar on X axis
                    br1 = np.arange(len(AB))
                    br2 = [x + barWidth for x in br1]
                    br3 = [x + barWidth for x in br2]
                    
                    # Make the plot
                    plt.bar(br1, AB, color ='b', width = barWidth,
                            edgecolor ='grey', label ='Alice-Bob')
                    plt.bar(br2, AE, color ='r', width = barWidth,
                            edgecolor ='grey', label ='Alice-Eve')
                    plt.bar(br3, BE, color ='m', width = barWidth,
                            edgecolor ='grey', label ='Bob-Eve')
                    
                    # Adding Xticks
                    plt.xlabel('RS(N,K)', fontweight ='bold', fontsize = 15)
                    plt.ylabel('Reconciliation Success Rate', fontweight ='bold', fontsize = 15)
                    plt.xticks([r + barWidth for r in range(len(AB))],
                            ["RS("+str(n1)+","+str(k)+")", "RS("+str(n2)+","+str(k)+")", "RS("+str(n3)+","+str(k)+")"])

                    plt.title("Average Reconciliation Success Rate with variations on RS(N,K) - Scenario 3", fontsize = 15)
                    plt.legend()
                    plt.show()

                    # Reconciliation Bar 1
                    barWidth = 0.25
                    fig = plt.subplots(figsize =(12, 8))

                    # set height of bar
                    AB = groupAverage(rec_rate_data11,batch_size)
                    AE = groupAverage(rec_rate_data21,batch_size)
                    BE = groupAverage(rec_rate_data31,batch_size)
                    
                    # Set position of bar on X axis
                    br1 = np.arange(len(AB))
                    br2 = [x + barWidth for x in br1]
                    br3 = [x + barWidth for x in br2]
                    
                    # Make the plot
                    plt.bar(br1, AB, color ='b', width = barWidth,
                            edgecolor ='grey', label ='Alice-Bob')
                    plt.bar(br2, AE, color ='r', width = barWidth,
                            edgecolor ='grey', label ='Alice-Eve')
                    plt.bar(br3, BE, color ='m', width = barWidth,
                            edgecolor ='grey', label ='Bob-Eve')
                    
                    # Adding Xticks
                    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
                    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
                    plt.xticks([r + barWidth for r in range(len(AB))],
                        np.arange(1, len(AB)+1)*batch_size)
                    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

                    plt.title("Reconciliation Success Rate Scenario 1 - RS("+str(n1)+","+str(k)+")")
                    plt.legend()
                    plt.show()

                    # Reconciliation Bar 2
                    barWidth = 0.25
                    fig = plt.subplots(figsize =(12, 8))

                    # set height of bar
                    AB = groupAverage(rec_rate_data12,batch_size)
                    AE = groupAverage(rec_rate_data22,batch_size)
                    BE = groupAverage(rec_rate_data32,batch_size)
                    
                    # Set position of bar on X axis
                    br1 = np.arange(len(AB))
                    br2 = [x + barWidth for x in br1]
                    br3 = [x + barWidth for x in br2]
                    
                    # Make the plot
                    plt.bar(br1, AB, color ='b', width = barWidth,
                            edgecolor ='grey', label ='Alice-Bob')
                    plt.bar(br2, AE, color ='r', width = barWidth,
                            edgecolor ='grey', label ='Alice-Eve')
                    plt.bar(br3, BE, color ='m', width = barWidth,
                            edgecolor ='grey', label ='Bob-Eve')
                    
                    # Adding Xticks
                    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
                    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
                    plt.xticks([r + barWidth for r in range(len(AB))],
                        np.arange(1, len(AB)+1)*batch_size)
                    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

                    plt.title("Reconciliation Success Rate Scenario 1 - RS("+str(n2)+","+str(k)+")")
                    plt.legend()
                    plt.show()


                    # Reconciliation Bar 3
                    barWidth = 0.25
                    fig = plt.subplots(figsize =(12, 8))

                    # set height of bar
                    AB = groupAverage(rec_rate_data13,1)
                    AE = groupAverage(rec_rate_data23,1)
                    BE = groupAverage(rec_rate_data33,1)
                    
                    # Set position of bar on X axis
                    br1 = np.arange(len(AB))
                    br2 = [x + barWidth for x in br1]
                    br3 = [x + barWidth for x in br2]
                    
                    # Make the plot
                    plt.bar(br1, AB, color ='b', width = barWidth,
                            edgecolor ='grey', label ='Alice-Bob')
                    plt.bar(br2, AE, color ='r', width = barWidth,
                            edgecolor ='grey', label ='Alice-Eve')
                    plt.bar(br3, BE, color ='m', width = barWidth,
                            edgecolor ='grey', label ='Bob-Eve')
                    
                    # Adding Xticks
                    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
                    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
                    plt.xticks([r + barWidth for r in range(len(AB))],
                        np.arange(1, len(AB)+1)*batch_size)
                    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

                    plt.title("Reconciliation Success Rate Scenario 1 - RS("+str(n3)+","+str(k)+")")
                    plt.legend()
                    plt.show()

                    if enable_flags['test_privacy_amplification']:
                        #Privacy Amplification
                        priv_amp_data = []
                        i = 0
                        rec_data1 = np.array(reconciliation11)
                        t_start = time.time()
                        while i < rec_data1.shape[0]:
                            priv_amp = privacyAmplification(rec_data1[i][1])
                            priv_amp_data.append(priv_amp)
                            i = i+1
                        t_end = time.time()
                        print("Time for privacy amplification: ", t_end-t_start)
                        #print(priv_amp_data)
                        if enable_flags['test_nist_rng']:
                            #NIST Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
                            priv_amp_bin_data = []

                            for P_A_data in priv_amp_data:
                                priv_amp_bin = str2arr(P_A_data)
                                priv_amp_bin_data.append(priv_amp_bin)

                            print(priv_amp_bin_data[0])

                            NIST_RNG_test(priv_amp_bin_data)