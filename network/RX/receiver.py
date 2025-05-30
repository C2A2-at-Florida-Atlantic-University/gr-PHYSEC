
import struct
import socket
import json

from flask import Flask, jsonify
import numpy as np  
if __name__ != '__main__':
    from RX.mpsk import MPSK
    from RX.pkt_rcv_gr38 import packetReceive
    from RX.sinusoid import Sinusoid
else:
    from mpsk import MPSK
    from pkt_rcv_gr38 import packetReceive
    from sinusoid import Sinusoid
import time
class Receiver():
    def __init__(self,
                gain,
                samp_rate,
                freq,
                bandwidth=20000000,
                buffer_size=0x800,
                SDR_ADDR="",
                UDP_port=40868,
                UDP_IP="127.0.0.1"):
        self.gain = gain
        self.samp_rate = samp_rate
        self.freq = freq
        self.bandwidth = bandwidth
        self.buffer_size =buffer_size
        self.SDR_ADDR = SDR_ADDR
        self.UDP_port=UDP_port
        self.UDP_IP=UDP_IP
        self.sock = self.set_UDP_socket()
        self.rx = None
        self.set_rx_data()
        print("Receiver Initialized")
        print("Gain: "+str(self.gain))
        print("Sampling Rate: "+str(self.samp_rate))
        print("Frequency: "+str(self.freq))
        print("Bandwidth: "+str(self.bandwidth))
        print("Buffer Size: "+str(self.buffer_size))
        print("SDR ID: "+str(self.SDR_ADDR))
        print("UDP Port: "+str(self.UDP_port))
        print("UDP IP: "+str(self.UDP_IP))

    def setFreq(self,freq):
        self.freq = freq
        self.rx.set_freq(self.freq)
        
    def setSamplingRate(self,samp_rate):
        self.samp_rate=samp_rate
        self.rx.set_samp_rate(self.samp_rate)
    
    def setGain(self,gain):
        self.gain=gain
        self.rx.set_gain(self.gain)
    
    def setBandwidth(self,bandwidth):
        self.bandwidth=bandwidth
        self.rx.set_bandwidth(self.bandwidth)
    
    def set_buffer_size(self,buffer_size):
        self.buffer_size=buffer_size
        self.rx.set_buffer_size(self.buffer_size)
        
    def set_rx_data(self):
        sps = 2 #symbols per sample
        del self.rx
        self.rx=packetReceive(
            samp_rate=self.samp_rate,
            sps=sps,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            UDP_port=self.UDP_port
        )
        
    def set_rx_MPSK(self,M):
        del self.rx
        self.rx=MPSK(
            samp_rate=self.samp_rate,
            sps=4,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            UDP_port=self.UDP_port,
            M=M
        )
        
    def set_rx_IQ(self):
        del self.rx
        self.rx=Sinusoid(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            UDP_port=self.UDP_port
        )
        
    #Set UDP_port=40860 for retrieving IQ
    def set_UDP_socket(self):
        sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.UDP_IP, self.UDP_port))
        sock.settimeout(0.5)
        return sock
    
    def data2IQ(self,data,bps=8):
        samples = []
        for i in range(0, len(data), bps):
            real, imag = struct.unpack('ff', data[i:i+bps])
            complex_num = complex(real, imag)
            samples.append(complex_num)
        return samples
    
    def retrieve_IQ(self,dataSize=2**16,samples=2**10):
        IQ_data=[]
        while len(IQ_data)<samples:
            try:
                data = self.retrieve_raw_data(dataSize)
                for i in range(0, len(data), 8):
                    real, imag=struct.unpack('ff', data[i:i+8])
                    complex_num=complex(real, imag)
                    IQ_data.append(complex_num)
            except socket.timeout:
                pass
        return IQ_data
    
    def clear_UDP_socket(self):
        print("clearing socket")
        try:
            while True:
                _, _ = self.sock.recvfrom(4096)
        except:  # noqa: E722
            pass
        print("done clearing socket")

    def retrieve_data(self,dataSize=8192):
        data, _ =self.sock.recvfrom(dataSize) 
        print(data)
        data=data.decode('ascii')
        data=json.loads(data)
        UUID=data['UUID']
        ts=data['ts']
        size=data['size']
        type=data['type']
        contents=data['data']
        return UUID,ts,size,type,contents

    def retrieve_raw_data(self,dataSize=8192):
        data, _ =self.sock.recvfrom(dataSize) 
        return data

    def start(self):
        self.clear_UDP_socket()
        self.rx.start()

    def stop(self):
        self.rx.stop()
        self.rx.wait()
        self.clear_UDP_socket()

def testReceiver():
    samp_rate=1e6
    gain=70
    freq=3.55e9

    print("Radio Setup Parameters")
    rx = Receiver(
        gain,
        samp_rate,
        freq,
        bandwidth=20000000,
        buffer_size=8192,
        SDR_ADDR="",
        UDP_port=40868,
        UDP_IP="127.0.0.1")
    
    rxType = input("Receiver Type [Data (1), IQ (2)]:")
    if rxType == "1":
        print("Setting data receiver")
        rx.set_rx_data()
    elif rxType == "2":
        print("Setting sinusoid receiver")
        rx.set_rx_IQ()
    else:
        print("no option "+rxType)
        return
    #receiver.set_rx_data()
    print("Starting RX")
    rx.clear_UDP_socket()
    rx.start()
    if rxType == "1":
        received = False
        while not(received):
            try:
                print("attempting to retrieving data")
                #UUID,ts,size,type,contents=rx.retrieve_data(dataSize=8192)
                contents=rx.retrieve_raw_data(dataSize=8192)
                received = True
            except Exception as err:
                print("Error:",err)
                pass
            time.sleep(2)
        print(contents)
    elif rxType == "2":
        received = False
        print("Retrieving UDP data")
        while not(received):
            try:
                data=rx.retrieve_IQ(samples=1024)
                received = True
            except Exception as error:
                print("An exception occurred:", error)
                pass
        # samples = rx.data2IQ(data)
        real_data = np.real(data)
        imag_data = np.imag(data)
        callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
        print("Callback:", callback)
        app = Flask(__name__)
        with app.app_context():
            response = jsonify(callback)
        print(response)
        # print(data)
        print(len(data))

    print("Stopping RX")
    rx.stop()
    rx.clear_UDP_socket()

if __name__ == '__main__':
    testReceiver()
    print("Done")