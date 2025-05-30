from flask import Flask, request, jsonify
import tensorflow as tf
from statistics import mean
import numpy as np
import unireedsolomon as rs
from nistrng import *
import hashlib
from tensorflow  import Graph, Session
from tensorflow.keras.models import load_model
from dataset_preparation import ChannelSpectrogram

'''global graph, model
graph = Graph()

model_path = '/home/siwn/datasets/'
feature_extractor_name = model_path+'QExtractor.h5'
print("loading model")
with tf.device('/CPU:0'):
    with graph.as_default():
        session = Session()
        with session.as_default():
            model = load_model(feature_extractor_name)'''

def convert_to_complex(data):
    '''Convert the loaded data to complex IQ samples.'''
    real = data[:8192]
    print("real data: ",real[0:10])
    imag = data[8192:]
    print("imag data: ",imag[0:10])
    complex_data = real + 1j*imag
    return complex_data

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

def privacyAmplification(data):
    # encode the string
    encoded_str = data.encode()
    # create sha3-256 hash objects
    obj_sha3_256 = hashlib.new("sha3_512", encoded_str)
    return(obj_sha3_256.hexdigest())

def create_spectrogram(IQ_data):
    print("IQ data: ",IQ_data[0:10])
    complex_data = convert_to_complex(np.array(IQ_data))
    print("complex data: ",complex_data[0:10])
    print(complex_data.shape)
    ChannelSpectrogramObj = ChannelSpectrogram()
    data = ChannelSpectrogramObj.channel_spectrogram(np.array([complex_data]),512)
    return data

def create_parity_bits(key,n=255,k=128):
    key = hex(int(arr2str(key), 2))
    key = str(key[2:]) #A hex key
    coder = rs.RSCoder(n,k)
    key_encode = coder.encode(key) #Encode A
    key_parity = key_encode[k:] #A parity bits
    return key_parity

def reconcile(key,parity,n=255,k=128):
    key = hex(int(arr2str(key), 2))
    key = str(key[2:]) #A hex key
    key_parity = key+parity #B key + A parity bits
    coder = rs.RSCoder(n,k)
    reconciled = False
    try:
        key_decode = coder.decode(key_parity) #Decode B key + A parity bits
        key_reconciled = key_decode[0] #Reconcilieted key
        reconciled = True
    except:
        key_reconciled = key
    return reconciled,key_reconciled

def feature_extraction(data):
        #with tf.device('/GPU:0'):
        print(data)
        model_path = '/home/siwn/datasets/'
        feature_extractor_name = model_path+'QExtractor.h5'
        print("loading model")
        with tf.device('/CPU:0'):
            model = load_model(feature_extractor_name)
        features = model.predict(data)
        return features

app=Flask(__name__)

'''@app.before_first_request
def get_model():
    global model
    model_path = '/home/siwn/datasets/'
    feature_extractor_name = model_path+'QExtractor.h5'
    print("loading model")
    with tf.device('/CPU:0'):
        model = load_model(feature_extractor_name)'''

@app.route('/data/createSpectrogram', methods=['POST'])
def createSpectrogram():
    print("createSpectrogram")
    data=request.get_json()
    IQ_data=data["IQ"]
    spectrogram = create_spectrogram(IQ_data)
    calback = {"spectrogram": spectrogram.tolist()}
    return jsonify(calback), 201

@app.route('/AI/extractFeatures', methods=['POST'])
def extractFeatures():
    print("extractFeatures")
    data=request.get_json()
    spectrogram=data["spectrogram"]
    features = feature_extraction(np.array(spectrogram))
    calback = {"features": features.tolist()}
    return jsonify(calback), 201

@app.route('/data/quantizeFeatures', methods=['POST'])
def quantizeFeatures():
    print("quantizeFeatures")
    data=request.get_json()
    features=data["features"]
    key = feature_quantization(features[0])
    calback = {"key": key}
    return jsonify(calback), 201

@app.route('/data/generateParityBits', methods=['POST'])
def generateParityBits():
    print("generateParityBits")
    data=request.get_json()
    key=data["key"]
    print(key[:10])
    parityBits = create_parity_bits(key)
    calback = {"parityBits": parityBits}
    return jsonify(calback), 201

@app.route('/data/reconcileKey', methods=['POST'])
def reconcileKey():
    print("reconcileKey")
    data=request.get_json()
    parityBits=data["parityBits"]
    key=data["key"]
    reconciled,key_reconciled = reconcile(key,parityBits)
    calback = {"reconciled":reconciled,"key_reconciled":key_reconciled}
    return jsonify(calback), 201

if __name__ == '__main__':
    
    app.run(host="127.0.0.1", port=5003, debug=True)