import random
import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import datetime as dt
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import os 
import psutil

#Dataset Generator for creating datasets
#Creates IQ datasets within the SigMF format
#IQ files as well as signal metadata
class sigMFDataset():
    def __init__(self):
        self.date_time = dt.datetime.utcnow().isoformat()+'Z'
        self.metadataIsSet = False
        
    def setData(self,data,label,samplesPerExample):
        self.data = data
        self.label = label
        self.SPE = samplesPerExample
        
    def createDataset(self):
        if self.metadataIsSet:
            self.createFolder()
            self.createIQFile()
            self.createMetadata()
        else:
            print("Set metadata first with setMetadata()")
    
    def createFolder(self):
        parent_dir = os.getcwd()
        directory = self.fileName+"_"+self.author+"_"+self.date_time
        self.path = os.path.join(parent_dir,directory)
        os.mkdir(self.path) 
        print("Directory '% s' created" % directory) 
    
    def createIQFile(self):
        self.data.tofile(self.fileName+'.sigmf-data')
    
    def setMetadata(self):
        # create the metadata
        self.fileName = input("File Name:")
        self.samp_rate = input("Sampling Rate:")
        self.freq = input("Sampling Frequency:")
        self.author = input("Author Email:")
        self.description = input("Description:")
        self.metadataIsSet = True
        
    def setMetadata(self,fileName,samp_rate,freq,author,description):
        # create the metadata
        self.fileName = fileName
        self.samp_rate = samp_rate
        self.freq = freq
        self.author = author
        self.description = description
        self.metadataIsSet = True
    
    def createMetadata(self):
        self.metadata = SigMFFile(
            data_file=self.fileName+'.sigmf-data', # extension is optional
            global_info = {
                SigMFFile.DATATYPE_KEY: get_data_type_str(self.data),  # in this case, 'cf32_le'
                SigMFFile.SAMPLE_RATE_KEY: self.samp_rate,
                SigMFFile.AUTHOR_KEY: self.author,
                SigMFFile.DESCRIPTION_KEY: self.description,
                SigMFFile.FREQUENCY_KEY: self.freq,
                SigMFFile.DATETIME_KEY: self.date_time,
            }
        )
        self.metadata.tofile(self.fileName+'.sigmf-meta')
        

def APILink(IP,port,path):
    url = "http://"+IP+":"+port+path 
    print("API URL:",url)
    return url 

def recordIQ(interfaceIP,IP,port,samples):
    path = "/rx/recordIQ"
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    data = {
        "samples": samples
    }
    response = session.post(url, data=json.dumps(data), headers=headers)
    response_json = response.json()
    imag = response_json["imag"]
    real = response_json["real"]
    return real,imag

def setRxIQ(interfaceIP,IP,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.get(url, data=json.dumps(data), headers=headers)
    return response
    
def setRxMPSK(interfaceIP,IP,port,M):
    path = "/rx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def setPHY(interfaceIP,IP,port,params):
    path = "/set/PHY"
    data = {
        "x": params["x"],
        "freq": params["freq"],
        "SamplingRate": params["SamplingRate"],
        "gain": params["gain"],
        "buffer_size": params["buffer_size"],
        "bandwidth": params["bandwidth"]
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response
    
def set_tx_sinusoid(interfaceIP,IP,port):
    path = "/tx/set/sinusoid"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def set_tx_MPSK(interfaceIP,IP,port,M):
    path = "/tx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def set_tx_pnSequence(interfaceIP,IP,port,sequence):
    path = "/tx/set/pnSequence"
    data = {
        "sequence": sequence
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def set_tx_fileSource(interfaceIP,IP,port,fileSource):
    path = "/tx/set/fileSource"
    data = {
        "fileSource": fileSource
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def start_tx(interfaceIP,IP,port):
    path = "/tx/start"
    data = {
        "message": "TX Start"
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response
    
def stop_tx(interfaceIP,IP,port):
    path = "/tx/stop"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    url = APILink(IP,port,path)
    session = setSessionInterface(interfaceIP)
    response = session.post(url, data=json.dumps(data), headers=headers)
    return response

def plotTimeDomain(I,Q,samples=-1,id=0):
    plt.plot(I[0:samples], color='red')
    plt.plot(Q[0:samples], color='blue')
    plt.xlabel('Time')
    plt.ylabel('IQ')
    plt.title('Time Domain Plot Node: '+str(id))
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.pause(1)
    plt.close()

def plotConstellationDiagram(I,Q,samples=-1,id=0):
    plt.scatter(I[0:samples], Q[0:samples], color='red')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Constellation Diagram Plot Node: '+str(id))
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.pause(1)
    plt.close()

def plotSpectrogram(I,Q,fs,samples=-1,id=0):
    x = np.array([complex(Q[i],I[i]) for i in range(len(I))])
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.title('Spectrogram Plot Node: '+str(id))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.pause(1)
    plt.close()
    
def RoundRecordData(sessions,params,type,tx,RX,samples=1024, metadata = {}):
    txInterface = next((sessions[sessionType]["interface"] for sessionType in sessions if tx in sessions[sessionType]["nodes"]), None)
    print("Setting TX")
    
    if type == "sinusoid":
        set_tx_sinusoid(txInterface,NodeIP[tx],port["radio"])
    elif type == "MPSK":
        set_tx_MPSK(txInterface,NodeIP[tx],port["radio"],metadata[type])
    elif type == "pnSequence":
        set_tx_pnSequence(txInterface,NodeIP[tx],port["radio"],metadata[type])
    elif type == "fileSource":
        set_tx_fileSource(txInterface,NodeIP[tx],port["radio"],metadata[type])
    time.sleep(0.5)
    setPHY(txInterface,NodeIP[tx],port["radio"],params["tx"])
    time.sleep(1)
    print("Starting TX")
    start_tx(txInterface,NodeIP[tx],port["radio"])
    time.sleep(2.5)
    data = {}
    print("Setting RX")
    for rx in RX:
        data[rx] = np.array([])
        rxInterface =  next((sessions[sessionType]["interface"] for sessionType in sessions if rx in sessions[sessionType]["nodes"]), None)
        print("RX Interface",rxInterface)
        setRxIQ(rxInterface,NodeIP[rx],port["radio"])
        time.sleep(0.5)
        setPHY(rxInterface,NodeIP[rx],port["radio"],params["rx"])
    print("Finished setting RX")
    time.sleep(1)    
    print("Recording Data")
    # for i in range(packets):
    # print("packet "+str(i)+" of "+str(packets))
    for rx in RX:
        rxInterface =  next((sessions[sessionType]["interface"] for sessionType in sessions if rx in sessions[sessionType]["nodes"]), None)
        print("RX Interface",rxInterface)
        inphase,quadrature = recordIQ(rxInterface,NodeIP[rx],port["radio"],samples)
        complexIQ = np.array(inphase)+np.array(quadrature)*1j
        data[rx] = np.append(data[rx],complexIQ)
        # plotTimeDomain(inphase,quadrature)
        # plotConstellationDiagram(inphase,quadrature)
        # plotSpectrogram(inphase,quadrature)
        # plotAvgPower(inphase,quadrature)
    print("Finished recording data")
    time.sleep(5)
    print("Stopping TX")
    stop_tx(txInterface,NodeIP[tx],port["radio"])
    return data

def RecordDataAmbient(params,RX,samples=1024):
    for rx in RX:
        print("start RX ",rx)
        response = setRxIQ(NodeIP[rx],port["radio"])
        print(response)
        inphase,quadrature = recordIQ(NodeIP[rx],port["radio"],samples)
        print("stop RX",rx)
        plotTimeDomain(inphase,quadrature,samples=1024)
        plotConstellationDiagram(inphase,quadrature,samples=1024)
    #return data
    
def formatData(Data,modulation):
    DS_node = np.array([])
    DS_labels = np.array([])
    DS_data = np.array([])
    for node in Data:
        for N in range(examples):
            DS_node = np.append(DS_node,node)
            DS_labels = np.append(DS_labels,modulation)
            complex_data = np.array(Data[node][N*samplesPerExample:(N+1)*samplesPerExample])
            imag_data = complex_data.imag
            real_data = complex_data.real
            real_imag_data = np.array([[real_data,  imag_data]])
            if len(DS_data) == 0: 
                DS_data = real_imag_data
            else:
                DS_data = np.append(DS_data,real_imag_data,axis=0)
            #print(DS_data.shape)

    return DS_data, DS_node, DS_labels

def recordDataSignalClassificationMs(TXs,RXs,params,Ms = [2,4],totalExamples=1):
    for tx in TXs:
        env = {"rx":RXs,"tx":tx}
        Ms = [2,4]
        for M in Ms:
            Data = RoundRecordData(params,type="MPSK",tx=env["tx"],RX=env["rx"],
                                    samples=samplesPerExample,packets=int(totalExamples),
                                    metadata={"MPSK":M})
            if M == 2:
                Modulation = "BPSK"
            elif M == 4:
                Modulation = "QPSK"    
            DS_data, DS_node, DS_labels = formatData(Data,Modulation)
            rxText = "_".join(str(rx) for rx in env["rx"])
            filename = Modulation+"_"+str(env["tx"])+"_TX_"+rxText+"_RX_20042024"
            file = h5py.File(filename+".h5", "w")
            # Create datasets within the HDF5 file
            file.create_dataset("data", data=DS_data)
            file.create_dataset("node", data=DS_node)
            file.close()
            
def recordDataSignalClassification(folder,sessions,TXs,RXs,params,modulations,totalSamples):
    datetime = dt.datetime.now().strftime("%Y%m%d")
    for tx in TXs:
        env = {"rx":RXs,"tx":tx}
        for modulation in modulations:
            print("Recording data for "+modulation+" TX=" +str(tx))
            fileSource = "Matlab/"+modulation+".dat"
            Data = RoundRecordData(sessions,params,type="fileSource",tx=env["tx"],RX=env["rx"],
                                    samples=totalSamples,metadata={"fileSource":fileSource})   
            DS_data, DS_node, DS_labels = formatData(Data,modulation)
            rxText = "_".join(str(rx) for rx in env["rx"])
            filename = modulation+"_"+str(env["tx"])+"_TX_"+rxText+"_RX_"+datetime
            file = h5py.File(folder+filename+".h5", "w")
            # Create datasets within the HDF5 file
            file.create_dataset("data", data=DS_data)
            file.create_dataset("node", data=DS_node)
            file.close()

    #Save json file containing all parameters for experiment
    json_object = json.dumps(params, indent=4)
    JSON_FILE_NAME = "Parameters_TX_"+"".join([str(tx)+"_" for tx in TXs])+"RX_"+"".join([str(rx)+"_" for rx in RXs])+"_"+str(len(modulations))+"Modulations_"+datetime+".json"
    with open(JSON_FILE_NAME, "w") as jsonFile:
        jsonFile.write(json_object)
            
def recordDataChannelFingerprinting(TXs,RXs,params,sequence = "glfsr",samples=1024,packets=1):
    for tx in TXs:
        print("TX:",tx)
        env = {"rx":RXs,"tx":tx}
        Data = RoundRecordData(params,type="pnSequence",tx=env["tx"],RX=env["rx"],samples=samples,packets=packets,M=1,sequence=sequence)
        DS_data, DS_node, DS_labels = formatData(Data,sequence)
        rxText = "_".join(str(rx) for rx in env["rx"])
        filename = "pnSequence"+"_"+str(env["tx"])+"_TX_"+rxText+"_RX_20042024"
        file = h5py.File(filename+".h5", "w")
        # Create datasets within the HDF5 file
        file.create_dataset("data", data=DS_data)
        file.create_dataset("node", data=DS_node)
        file.close()

# Calculate the average power of the signal in DB given the real and imaginary samples
def calculate_power(real, imag):
    power = np.mean(real**2 + imag**2)
    power_db = 10*np.log10(power)
    return power_db

# function getting IQ samples from .dat files:
def get_complex_signal(file):
    # Read the .dat file as float32
    data = np.fromfile(file, dtype=np.float32)
    # Convert to complex IQ samples
    complex_signal = data[::2] + 1j * data[1::2]
    return complex_signal

class BoundAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, source_ip):
        self.source_ip = source_ip
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        # Bind the HTTP requests to a specific source IP (interface)
        kwargs['source_address'] = (self.source_ip, 0)
        super().init_poolmanager(*args, **kwargs)

def setSessionInterface(interfaceIP):
    # Create a requests session
    session = requests.Session()
    # Create a custom HTTP adapter that binds to the interface IP
    adapter = BoundAdapter(interfaceIP)
    # Mount the adapter for HTTP and HTTPS traffic
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Now requests will always use the specified interface
    return session

def getInterfaces():
    myInterfaces = {"WIFI":{"name":"en0","address":None},"ETH":{"name":"en17","address":None}}
    interfaces = psutil.net_if_addrs()
    for interface, addresses in interfaces.items():
        for interfaceName in myInterfaces:
            if interface == myInterfaces[interfaceName]["name"]:
                for addr in addresses:
                    if addr.family.name == "AF_INET":
                        myInterfaces[interfaceName]["address"] = addr.address
                        print(f"For {interfaceName} IP is {myInterfaces[interfaceName]['address']}")
    return myInterfaces

NodeIP = {1:"10.15.7.109",2:"10.13.251.58",3:"10.15.6.201",4:"10.15.6.243",5:"10.15.7.42", 
        6:"10.15.6.180",7:"10.15.7.12",8:"10.15.7.52",9:"127.0.0.1",10:"127.0.0.1"}

port = {'orch':'5001','radio':'5002','ai':'5003'}

examples= 10000
samplesPerExample = 1024
# samplesPerPacket = samplesPerExample*examples
freq = 2.29e9
samp_rate = 600e3
gainRX = 60
gainTX = 0
bufferSize = 0x800
totalSamples = examples * samplesPerExample 
print("Total_samples:",totalSamples)
# examplesPerPacket = samplesPerPacket/samplesPerExample   #I_Sig,Q_Sig
# print("examplesPerPacket:",examplesPerPacket)
# packetsNeeded = int(math.ceil(examples/examplesPerPacket))
# print("packetsNeeded:",packetsNeeded)
paramsTx = {"x":"tx","freq":freq,"SamplingRate":samp_rate,"gain":gainTX,"buffer_size":bufferSize,"bandwidth":samp_rate*2}
paramsRx = {"x":"rx","freq":paramsTx["freq"],"SamplingRate":int(paramsTx["SamplingRate"]*2),"gain":gainRX,"buffer_size":bufferSize,"bandwidth":paramsTx["SamplingRate"]*2*2}
params = {"tx":paramsTx,"rx":paramsRx}

# modulations = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "B-FM", "DSB-AM", "SSB-AM"]
modulations = ["BPSK"]

# Define the source IPs for each network interface
myInterfaces = getInterfaces()
interfaceSessions = {"wifi": {"interface":myInterfaces["WIFI"]["address"],"nodes":[2]}, "eth":{"interface":myInterfaces["ETH"]["address"],"nodes":[1,3,4,5,6]}}

TXs = [1]
RXs = [3,4,5,6]
folder = "/Users/josea/Workspaces/siwn/data/"
recordSignalClassification = 1
recordChannelFingerprinting = 0
recordAmbient = 0

if recordSignalClassification:
    # Record data for signal classification
    recordDataSignalClassification(folder,interfaceSessions,TXs,RXs,params,modulations,totalSamples)
    #recordDataSignalClassificationMs(TXs,RXs,params,Ms = [2,4])
if recordChannelFingerprinting:
    # Record data for channel fingerprinting
    recordDataChannelFingerprinting(TXs,RXs,params,sequence="glfsr",samples=totalSamples,packets=1)
if recordAmbient:
    # Record ambient data
    RecordDataAmbient(params,RXs,samples=totalSamples)

analyzeSignals = 0
if analyzeSignals:
    # Analyze recorded signals
    # Get names of all files from the folder
    files = [f for f in os.listdir(folder) if f.endswith('.h5')]
    print('Files: ' + str(files))
    # Number of files
    numRows = len(files)
    pltTimeDomain = 1
    pltFrequencyDomain = 1
    pltSpectrogram = 1

    # Create subplots (3 rows per file: signal, spectrogram, and frequency domain)
    numColumns = pltTimeDomain+pltSpectrogram+pltFrequencyDomain
    fig, axes = plt.subplots(numRows, numColumns, figsize=(10*numColumns, 10*numRows))
    signalPowers = []
    # Loop through each file
    for i, file in enumerate(files):
        print("File:",file)
        k = 0
        # Get the complex signal from the file
        with h5py.File(folder+file, 'r') as f:
            # Load the data and labels datasets.
            data = f['data'][:]
            print("data Shape:",data.shape)
        complex_signal = data[0][0] + 1j * data[0][1]
        real = np.real(complex_signal)
        imag = np.imag(complex_signal)
        print("Signal:",complex_signal)
        print("Signal Shape:",complex_signal.shape)
        # complex_signal = get_complex_signal(folder+file)
        power = calculate_power(np.real(complex_signal), np.imag(complex_signal))
        signalPowers.append(power)
        print("Signal Length:",len(complex_signal))
        print("Power:",power)
        if pltTimeDomain:
            # Plot Time-domain Signal 
            axes[i, k].plot(real, label="Real")
            axes[i, k].plot(imag, label="Imaginary")
            axes[i, k].set_title(f"Time-Domain Signal - {file}")
            axes[i, k].set_xlabel("Samples")
            axes[i, k].set_ylabel("Amplitude")
            axes[i, k].legend()
            k+=1
        if pltFrequencyDomain:
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
        if pltSpectrogram:
            # Plot Spectrogram
            x = np.array([complex_signal[i] for i in range(len(complex_signal))])
            f, t, Sxx = signal.spectrogram(x, 1e6)
            axes[i, k].set_title(f"Spectrogram - {file}")
            axes[i, k].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
            axes[i, k].set_ylabel('Frequency [Hz]')
            axes[i, k].set_xlabel('Time [sec]')
            k+=1
            
    # Adjust layout
    plt.tight_layout()
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
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('power.png')
