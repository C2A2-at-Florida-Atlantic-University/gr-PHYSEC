import requests
import json
import time
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
    real = [x * 1000 for x in real]
    imag = [x * 1000 for x in imag]
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
    listen_request_send_sinusoid()
    time.sleep(2)
    print("sending request and getting IQ data")
    IQ_data = send_request_get_IQ()
    print(len(IQ_data))
    #print(IQ_data)
    print("create spectrogram")
    spectrogram = create_spectrogram(IQ_data)
    #print(spectrogram)
    print("extract features")
    features = extractFeatures(spectrogram)
    #print(features)
    print("quantizing features")
    key = quantizeFeatures(features)
    print(key)
    print("generate encoded parity bits")
    parityBits = generateParityBits(key)
    print("Transforming encoded values into hex")
    byte_data = parityBits.encode('latin1') 
    hex_representation = byte_data.hex()
    print(parityBits)
    print("sending parity bits")
    send_data(hex_representation)
    #Send Parity Bits
    print("Get reconciliation result")
    UUID,contents,received,size,ts,type = listen_channel()
    print(contents)
    '''if contents == "False":
        time.sleep(10)
        keyGeneration_Alice(n)'''

def keyGeneration_Bob(n=0):
    n = n+1
    print(n)
    print("sending request and getting IQ data")
    IQ_data = send_request_get_IQ()
    print(len(IQ_data))
    #print(IQ_data)
    time.sleep(2)
    print("receiving request and sending sinusoid")
    listen_request_send_sinusoid()
    print("create spectrogram")
    spectrogram = create_spectrogram(IQ_data)
    #print(spectrogram)
    print("extract features")
    features = extractFeatures(spectrogram)
    #print(features)
    print("quantizing features")
    key = quantizeFeatures(features)
    print(key)
    print("getting parity bits")
    #Get Parity Bits
    UUID,contents,received,size,ts,type = listen_channel()
    print(contents)
    #Transforming hex values into original encoded values
    hex_representation = contents
    byte_data_from_hex = bytes.fromhex(hex_representation)
    parityBits = byte_data_from_hex.decode('latin1')
    print("reconciling key")
    reconciled,key_reconciled = reconcileKey(key,parityBits)
    print(reconciled)
    print(key_reconciled)
    print("Sending reconciliation results")
    send_data(str(reconciled))

    '''if not(reconciled):
        time.sleep(10)
        keyGeneration_Bob(n)'''
        

#keyGeneration_Bob()
keyGeneration_Alice()