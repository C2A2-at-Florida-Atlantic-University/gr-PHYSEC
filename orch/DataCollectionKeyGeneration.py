import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py

def APILink(IP,port,path):
    return "http://"+IP+":"+port+path    

def recordIQ(nodeID,port,samples):
    path = "/rx/recordIQ"
    data = {
        "samples": samples
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    # response_rx = requests.get(APILink(NodeIPs[nodeID],port,path))
    # print("Response:",response)
    response_json = response.json()
    # print("Response JSON:",response_json)
    imag = response_json["imag"]
    real = response_json["real"]
    return real,imag

def setRxIQ(nodeID,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.get(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def setPHY(nodeID,port,params):
    path = "/set/PHY"
    data = {
        "x": params["x"],
        "freq": params["freq"],
        "SamplingRate": params["SamplingRate"],
        "gain": params["gain"][nodeID][params["x"]]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response
    
def set_tx_sinusoid(nodeID,port):
    path = "/tx/set/sinusoid"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_MPSK(nodeID,port,M):
    path = "/tx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_pnSequence(nodeID,port,sequence):
    path = "/tx/set/pnSequence"
    data = {
        "sequence": sequence
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def start_tx(nodeID,port):
    path = "/tx/start"
    data = {
        "message": "TX Start"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response
    
def stop_tx(nodeID,port):
    path = "/tx/stop"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
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
    # plt.show()
    # Show for 0.5 seconds
    plt.pause(0.5)
    plt.clf()  # Clear the figure for the next plot

def setTXNode(params,type,nodeID,metadata = {"pnSequence":"glfsr"}):
    print("type:",type)
    if type == "sinusoid":
        response = set_tx_sinusoid(nodeID,port["radio"])
    elif type == "pnSequence":
        response = set_tx_pnSequence(nodeID,port["radio"],metadata[type])
    response = setPHY(nodeID,port["radio"],params["tx"])
    response = start_tx(nodeID,port["radio"])
    
def setRXNode(params,nodeID):
    response = setRxIQ(nodeID,port["radio"])
    response = setPHY(nodeID,port["radio"],params["rx"])
    
def RecordNodeData(nodeID,samples):
    return recordIQ(nodeID,port["radio"],samples)
    
def stopTXNode(nodeID):
    response = stop_tx(nodeID,port["radio"])

def setupNodesPingPong(RX1,RX2,TX,params,type,metadata = {"pnSequence":"glfsr"}):
    setRXNode(params,TX)    
    setRXNode(params,RX1)
    setRXNode(params,RX2)
    setTXNode(params,type,TX,metadata)
    
def collect_data_ping_pong_3Nodes(params, nodes, packages, type, channel_Labels = [1,2,3],metadata = {"pnSequence":"glfsr"}):
    i = 1
    # IQ_Samples = np.array([])
    I = []
    Q = []
    channel = []
    instance = []
    ids = []
    tx = []
    rx = []
    timestamp = []
    id = 0
    timeSleep = 0.3
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    numberOfSamples = 8192
    print("Setting RX Node: ", Eve)
    
    generatePlots = True
    while i<packages*2+1:
        print(i)
        if i%2 == 0:
            print("Setting TX Node: ", Bob)
            setTXNode(params,type,Bob,metadata)
            
            print("Setting RX Node: ", Alice)
            setRXNode(params, Alice)
            setRXNode(params,Eve)
            time.sleep(timeSleep)
            
            print("Recording RX Node: ", Alice)
            real1, imaginary1 = RecordNodeData(Alice, samples=numberOfSamples)
            idxStart1 = len(real1)-numberOfSamples
            I.append(real1[idxStart1:])
            Q.append(imaginary1[idxStart1:])
            channel.append(channel_Labels[0])
            instance.append(1)
            ids.append(id)
            tx.append(Bob)
            rx.append(Alice)
            timestamp.append(int(time.time()))
            
            print("Recording RX Node: ", Eve)
            real2, imaginary2 = RecordNodeData(Eve, samples=numberOfSamples)
            idxStart2 = len(real2)-numberOfSamples
            I.append(real2[idxStart2:])
            Q.append(imaginary2[idxStart2:])
            channel.append(channel_Labels[1])  
            instance.append(2)   
            ids.append(id)
            tx.append(Bob)
            rx.append(Eve)
            timestamp.append(int(time.time()))
            
            if(generatePlots):
                plotTimeDomain(
                    real1[idxStart1:], 
                    imaginary1[idxStart1:], 
                    samples=numberOfSamples, id=Alice)
                plotTimeDomain(
                    real2[idxStart2:], 
                    imaginary2[idxStart2:], 
                    samples=numberOfSamples, id=Eve
                )
            
            stopTXNode(Bob)
            time.sleep(timeSleep)

        else:
            id = id + 1
            print(("Setting TX Node: ", Alice))
            setTXNode(params,type,Alice,metadata)
            print("Setting RX Node: ", Bob)
            setRXNode(params,Bob)
            setRXNode(params,Eve)
            time.sleep(timeSleep)
            
            print("Recording RX Node: ", Bob)
            real1, imaginary1 = RecordNodeData(Bob, samples=numberOfSamples)
            idxStart1 = len(real1)-numberOfSamples
            I.append(real1[idxStart1:])
            Q.append(imaginary1[idxStart1:])
            channel.append(channel_Labels[0])
            instance.append(3)
            ids.append(id)
            tx.append(Alice)
            rx.append(Bob)
            timestamp.append(int(time.time()))

            print("Recording RX Node: ", Eve)
            real2, imaginary2 = RecordNodeData(Eve, samples=numberOfSamples)
            idxStart2 = len(real2)-numberOfSamples
            I.append(real2[idxStart2:])
            Q.append(imaginary2[idxStart2:])
            channel.append(channel_Labels[2]) # Changed from "labels" to "channel"
            instance.append(4)
            ids.append(id)
            tx.append(Alice)
            rx.append(Eve)
            timestamp.append(int(time.time()))

            stopTXNode(Alice)
            
            if(generatePlots):
                plotTimeDomain(
                    real1[idxStart1:], 
                    imaginary1[idxStart1:], 
                    samples=numberOfSamples, id=Bob)
                plotTimeDomain(
                    real2[idxStart2:], 
                    imaginary2[idxStart2:], 
                    samples=numberOfSamples, id=Eve
                )
            time.sleep(timeSleep)
        
        i = i + 1
    return I, Q, channel, instance, ids, tx, rx, timestamp

def create_dataset(filename, I, Q, channel, instance, ids, tx, rx, timestamp):
    with h5py.File(filename, "w") as data_file:
        dset = data_file.create_dataset("I", data=I)
        dset = data_file.create_dataset("Q", data=Q)
        dset = data_file.create_dataset("ids", data=[ids])
        dset = data_file.create_dataset("instance", data=[instance])
        dset = data_file.create_dataset("channel", data=[channel])
        dset = data_file.create_dataset("tx", data=[tx])
        dset = data_file.create_dataset("rx", data=[rx])
        dset = data_file.create_dataset("timestamp", data=[timestamp])
    # Save dataset to file
    print("Dataset saved to", filename)

def loadOTALabConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":70}
            }
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"ota-nuc1.emulab.net",    # b210 nuc node 1
        2:"ota-nuc2.emulab.net",    # b210 nuc node 2
        3:"ota-nuc3.emulab.net",    # b210 nuc node 3
        4:"ota-nuc4.emulab.net",    # b210 nuc node 4
        5:"pc783.emulab.net",       # x310 radio node 1
        6:"pc792.emulab.net",       # x310 radio node 2
        7:"pc796.emulab.net",       # x310 radio node 3
        8:"pc781.emulab.net"        # x310 radio node 4
    }
    NodeGains = {
        1:gainConfigs["b210"],
        2:gainConfigs["b210"],
        3:gainConfigs["b210"],
        4:gainConfigs["b210"],
        5:gainConfigs["x310"],
        6:gainConfigs["x310"],
        7:gainConfigs["x310"],
        8:gainConfigs["x310"]
    }
    NodeConfigs = [
        # Create every triplet combination for nodes 1,2,3,4,5,6,7,8
        [1,2,3],  # b210, b210, b210
        [2,4,3],  # b210, b210, b210
        [4,2,8],  # b210, b210, x310
        [4,2,5],  # b210, b210, x310
        [1,4,5],  # b210, b210, x310
        [1,4,8],  # b210, b210, x310
        [5,7,8],  # x310, x310, x310
        [5,8,7],  # x310, x310, x310
        [8,5,4],  # x310, x310, b210
        [8,5,1],  # x310, x310, b210
        [8,4,1],  # x310, b210, b210
        [4,8,5]   # b210, x310, x310
    ]
    return NodeIPs, NodeGains, NodeConfigs

def loadOTADenseConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":70}
            }
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"cnode-ebc.emulab.net",       # EBC dense node with b210
        2:"cnode-guesthouse.emulab.net",# Guesthouse dense node with b210
        3:"cnode-moran.emulab.net",     # Moran dense node with b210
        4:"cnode-ustar.emulab.net",     # Ustar dense node with b210
        5:"localhost",                  # Local computer with b210
        6:"162.168.10.101",             # Jetson nano 1 with b210
        7:"162.168.10.102"              # Jetson nano 2 with b210
    }
    NodeGains = {
        1:gainConfigs["b210"],
        2:gainConfigs["b210"],
        3:gainConfigs["b210"],
        4:gainConfigs["b210"],
        5:gainConfigs["b210"],
        6:gainConfigs["b210"],
        7:gainConfigs["b210"]
    }
    NodeConfigs = [
        # [1,2,3],  # EBC, Guesthouse, Moran
        # [2,3,1],  # Guesthouse, Moran, EBC
        # [1,3,2],  # EBC, Moran, Guesthouse
        # [4,3,1],  # Ustar, Moran, EBC
        # [4,3,2],  # Ustar, Moran, Guesthouse
        # [4,1,3],  # Ustar, EBC, Moran
        # [4, 3, 5],  # Ustar, Moran, Local
        [1,2,5]
    ]
    return NodeIPs, NodeGains, NodeConfigs

def loadOTARooftopConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":70}
            }
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"cnode-ebc.emulab.net",       # EBC dense node with b210
        2:"cnode-guesthouse.emulab.net",# Guesthouse dense node with b210
        3:"cnode-moran.emulab.net",     # Moran dense node with b210
        4:"cnode-ustar.emulab.net",     # Ustar dense node with b210
        5:"localhost",                  # Local computer with b210
        6:"162.168.10.101",             # Jetson nano 1 with b210
        7:"162.168.10.102"              # Jetson nano 2 with b210
    }
    NodeGains = {
        1:gainConfigs["x310"],
        2:gainConfigs["x310"],
        3:gainConfigs["x310"],
        4:gainConfigs["x310"],
        5:gainConfigs["x310"],
        6:gainConfigs["x310"],
        7:gainConfigs["x310"]
    }
    NodeConfigs = [
        # [1,2,3],  # EBC, Guesthouse, Moran
        # [2,3,1],  # Guesthouse, Moran, EBC
        # [1,3,2],  # EBC, Moran, Guesthouse
        # [4,3,1],  # Ustar, Moran, EBC
        # [4,3,2],  # Ustar, Moran, Guesthouse
        # [4,1,3],  # Ustar, EBC, Moran
        # [4, 3, 5],  # Ustar, Moran, Local
        [1,2,5]
    ]
    return NodeIPs, NodeGains, NodeConfigs

if __name__ == "__main__":

    # NodeIPs, NodeGains, nodeConfigs = loadOTALabConfig()
    NodeIPs, NodeGains, nodeConfigs = loadOTADenseConfig()

    port = {'orch':'5001','radio':'5002','ai':'5003'}

    examples= 100
    freq = 3.450e9
    samp_rate = 600e3

    paramsTx = {
        "x":"tx",
        "freq":freq,
        "SamplingRate":samp_rate,
        "gain":NodeGains
    }
    paramsRx = {
        "x":"rx",
        "freq":paramsTx["freq"],
        "SamplingRate":int(paramsTx["SamplingRate"]*2),
        "gain":NodeGains
    }
    params = {"tx":paramsTx,"rx":paramsRx}

    type = "sinusoid" #pnSequence, MPSK, sinusoid
        
    for nodes in nodeConfigs:
        print(f"Collecting data for node config: Alice={nodes[0]}, Bob={nodes[1]}, Eve={nodes[2]}")
        I, Q, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes(
                                            params, 
                                            nodes, 
                                            examples, 
                                            type
                                        )
        I_arr = np.array(I)
        Q_arr = np.array(Q)
        channel_arr = np.array(channel)
        instance_arr = np.array(instance)
        ids_arr = np.array(ids)
        tx_arr = np.array(tx)
        rx_arr = np.array(rx)
        timestamp_arr = np.array(timestamp)
        ts = int(time.time())
        create_dataset(
            "Dataset_OTADense_Channels_"+type+"_"+str(examples)+"_"+"".join(str(node) for node in nodes)+"_"+str(ts)+".hdf5",
            I_arr,
            Q_arr,
            channel_arr, 
            instance_arr, 
            ids_arr,
            tx_arr,
            rx_arr,
            timestamp_arr
        )

    # ###############

    # nodes = [2,5,4]

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[0],nodes[1],nodes[2]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[2],nodes[0],nodes[1]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[1],nodes[2],nodes[2]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)