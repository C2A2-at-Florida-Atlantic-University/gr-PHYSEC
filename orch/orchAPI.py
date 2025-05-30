import requests
import json
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt

radioAPI = "http://127.0.0.1:5002"
aiAPI = "http://127.0.0.1:5003"

def arr2str(arr):
    str_arr = ''
    for i in arr:
        str_arr += str(i)
    return str_arr

def create_spectrogram(IQ_data):
    data = {
        "IQ": IQ_data
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(aiAPI+"/data/createSpectrogram", data=json.dumps(data), headers=headers)
    response = response.json()
    spectrogram = response["spectrogram"]
    return spectrogram

def extractFeatures(spectrogram):
    data = {
        "spectrogram": spectrogram
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(aiAPI+"/AI/extractFeatures", data=json.dumps(data), headers=headers)
    response = response.json()
    features = response["features"]
    return features

def quantizeFeatures(features):
    data = {
        "features": features
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(aiAPI+"/data/quantizeFeatures", data=json.dumps(data), headers=headers)
    response = response.json()
    key = response["key"]
    return key

def generateParityBits(key):
    data = {
        "key": key
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(aiAPI+"/data/generateParityBits", data=json.dumps(data), headers=headers)
    response = response.json()
    parityBits = response["parityBits"]
    return parityBits

def reconcileKey(key,parityBits):
    data = {
        "key": key,
        "parityBits":parityBits
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(aiAPI+"/data/reconcileKey", data=json.dumps(data), headers=headers)
    response = response.json()
    reconciled = response["reconciled"]
    key_reconciled = response["key_reconciled"]
    return reconciled,key_reconciled

def request_CFF():
    data = {
        "message": "request"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(radioAPI+"/tx/data", data=json.dumps(data), headers=headers)
    return response

def send_data(data):
    data = {
        "message": data
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(radioAPI+"/tx/data", data=json.dumps(data), headers=headers)
    return response

def send_sinusoid():
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(radioAPI+"/tx/sinusoid", data=json.dumps(data), headers=headers)
    return response

def listen_channel():
    response_rx = requests.get(radioAPI+"/rx/data")
    response = response_rx.json()
    UUID = response["UUID"]
    contents = response["contents"]
    received = response["received"]
    size = response["size"]
    ts = response["ts"]
    type = response["type"]
    return UUID,contents,received,size,ts,type

def get_parity_bits():
    response_rx = requests.get(radioAPI+"/rx/raw_data")
    response = response_rx.json()
    received = response["received"]
    data = response["data"]
    print(data)
    return received,data

def get_IQ():
    response_rx = requests.get(radioAPI+"/rx/sinusoid")
    response_json = response_rx.json()
    imag = response_json["imag"]
    real = response_json["real"]
    return real,imag

def listen_request_send_sinusoid():
    UUID,contents,received,size,ts,type = listen_channel()
    if contents == "request":
        send_sinusoid()
    else:
        listen_request_send_sinusoid()
    
def send_request_get_IQ():
    request_CFF()
    time.sleep(5)
    real,imag = get_IQ()
    real = real[0:8192]
    imag = imag[0:8192]
    #real = [x * 1000 for x in real]
    #imag = [x * 1000 for x in imag]
    plt.plot(real, color='red')
    plt.plot(imag, color='blue')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Complex Data Plot')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()
    data = real + imag
    return data

def keyGeneration_Alice(n=0):
    n = n+1
    print(n)
    print("receiving request and sending sinusoid")
    start_time = time.time()
    listen_request_send_sinusoid()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for receiving request and sending sinusoid =",elapsed_time)
    time.sleep(2)
    print("sending request and getting IQ data")
    start_time = time.time()
    IQ_data = send_request_get_IQ()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for sending request and getting IQ data =",elapsed_time)
    print(len(IQ_data))
    #print(IQ_data)
    print("create spectrogram")
    start_time = time.time()
    spectrogram = create_spectrogram(IQ_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for signal processing =",elapsed_time)
    #print(spectrogram)
    print("extract features")
    start_time = time.time()
    features = extractFeatures(spectrogram)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for extract features =",elapsed_time)
    #print(features)
    print("quantizing features")
    start_time = time.time()
    key = quantizeFeatures(features)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for quantizing features =",elapsed_time)
    print(key)
    print("generate encoded parity bits")
    start_time = time.time()
    parityBits = generateParityBits(key)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for generate encoded parity bits =",elapsed_time)
    print("Transforming encoded values into hex")
    byte_data = parityBits.encode('latin1') 
    hex_representation = byte_data.hex()
    print(parityBits)
    print("sending parity bits")
    start_time = time.time()
    send_data(hex_representation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for sending parity bits =",elapsed_time)
    #Send Parity Bits
    print("Get reconciliation result")
    start_time = time.time()
    UUID,contents,received,size,ts,type = listen_channel()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for Get reconciliation result =",elapsed_time)
    print(contents)
    if contents == "False":
        time.sleep(10)
        keyGeneration_Alice(n)


def keyGeneration_Bob(n=0):
    n = n+1
    print(n)
    print("sending request and getting IQ data")
    start_time = time.time()
    IQ_data = send_request_get_IQ()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for sending request and getting IQ data =",elapsed_time)
    print(len(IQ_data))
    #print(IQ_data)
    time.sleep(2)
    print("receiving request and sending sinusoid")
    start_time = time.time()
    listen_request_send_sinusoid()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for receiving request and sending sinusoid IQ data =",elapsed_time)
    print("create spectrogram")
    start_time = time.time()
    spectrogram = create_spectrogram(IQ_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for Signal Processing =",elapsed_time)
    #print(spectrogram)
    print("extract features")
    start_time = time.time()
    features = extractFeatures(spectrogram)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for extracting features =",elapsed_time)
    #print(features)
    print("quantizing features")
    start_time = time.time()
    key = quantizeFeatures(features)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for quantizing features =",elapsed_time)
    print(key)
    print("getting parity bits")
    #Get Parity Bits
    start_time = time.time()
    UUID,contents,received,size,ts,type = listen_channel()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for receiving parity bits =",elapsed_time)
    print(contents)
    #Transforming hex values into original encoded values
    hex_representation = contents
    byte_data_from_hex = bytes.fromhex(hex_representation)
    parityBits = byte_data_from_hex.decode('latin1')
    print("reconciling key")
    start_time = time.time()
    reconciled,key_reconciled = reconcileKey(key,parityBits)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for reconciling key =",elapsed_time)
    print(reconciled)
    print(key_reconciled)
    print("Sending reconciliation results")
    start_time = time.time()
    send_data(str(reconciled))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for Sending reconciliation results =",elapsed_time)

    if not(reconciled):
        time.sleep(10)
        keyGeneration_Bob(n)



#keyGeneration_Bob()
keyGeneration_Alice()
