from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
import numpy as np

phy = None

class phyAPI(FlaskView):
    def __init__(self):
        self.app = Flask(__name__)
    
    def start(self,port,ip):
        phyAPI.register(self.app, route_base='/')
        self.app.run(host=ip, port=port)
    
    def injectNode(self, injectedNode):
        global phy
        phy = injectedNode

    @route('/tx/data', methods=['POST'])
    def tx_data(self):
        #{"mode":"data","data":"Hello","time":1}
        data=request.get_json()
        message=data["message"]
        phy.send_data_wait_confirmation(data=message)
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/tx/sinusoid', methods=['POST'])
    def tx_sinusoid(self):
        #{"mode":"data","data":"Hello","time":1}
        phy.transmit_sinusoid_await_confirmation()
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/tx/set/sinusoid', methods=['POST'])
    def tx_set_sinusoid(self):
        phy.setTxSinusoid()
        callback = {"contents": "setTxSinusoid" }
        return jsonify(callback), 201

    @route('/tx/set/MPSK', methods=['POST'])
    def tx_set_MPSK(self):
        data=request.get_json()
        M=data["M"]
        phy.setTxMPSK(M)
        callback = {"contents": "setTxMPSK" }
        return jsonify(callback), 201
    
    @route('/tx/set/pnSequence', methods=['POST'])
    def tx_set_pnSequence(self):
        data=request.get_json()
        sequence=data["sequence"]
        phy.setTxPnSequence(sequence)
        callback = {"contents": "setTxPnSequence" }
        return jsonify(callback), 201
    
    @route('/tx/set/fileSource', methods=['POST'])
    def tx_set_fileSource(self):
        data=request.get_json()
        fileSource=data["fileSource"]
        phy.setTxFileSource(fileSource)
        callback = {"contents": "setTxFileSource" }
        return jsonify(callback), 201

    @route('/tx/start', methods=['POST'])
    def tx_start(self):
        phy.transmitter.start()
        callback = {"contents": "transmitting" }
        return jsonify(callback), 201
    
    @route('/tx/stop', methods=['POST'])
    def tx_stop(self):
        phy.transmitter.stop()
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/rx/data', methods=['GET'])
    def rx_data(self):
        # Replace with your logic to retrieve items
        received,UUID,ts,size,type,contents = phy.receive_data_send_confirmation()
        callback = {"received":received,"UUID":UUID,"ts":ts,"size":size,"type":type,"contents": contents }
        return jsonify(callback), 200

    @route('/rx/raw_data', methods=['GET'])
    def raw_data(self):
        # Replace with your logic to retrieve items
        received,data = phy.receive_raw_data_send_confirmation()
        #print(data)
        callback = {"received":received,"data":data}
        return jsonify(callback), 200

    @route('/rx/sinusoid', methods=['GET'])
    def rx_sinusoid(self):
        IQ_data = phy.receive_sinusoid_send_confirmation()
        real_data = np.real(IQ_data)
        imag_data = np.imag(IQ_data)
        callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
        return jsonify(callback), 200

    @route('/rx/set/IQ', methods=['GET'])
    def rx_setIQ(self):
        contents = phy.set_receive_IQ()
        callback = {"contents": contents}
        return jsonify(callback), 200

    @route('/rx/set/MPSK', methods=['POST'])
    def rx_setMPSK(self):
        data=request.get_json()
        M=data["M"]
        phy.set_receive_MPSK(M)
        callback = {"contents": "done"}
        return jsonify(callback), 200

    @route('/rx/recordIQ', methods=['POST'])
    def rx_recordIQ(self):
        try:
            data=request.get_json()
            phy.receiver.start()
            IQ_data = phy.receiver.retrieve_IQ(samples=data["samples"])
            phy.receiver.stop()
            real_data = np.real(IQ_data)
            imag_data = np.imag(IQ_data)
            callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
        except Exception as error:
            callback = {"error": str(error)}
        return jsonify(callback), 200
    
    @route('/set/PHY', methods=['POST'])
    def setTx(self):
        data=request.get_json()
        if "freq" in data:
            phy.setFreq(data["freq"],data["x"])
        if "SamplingRate" in data:
            phy.setSamplingRate(data["SamplingRate"],data["x"])
        if "gain" in data:
            phy.setGain(data["gain"],data["x"])
        if "bandwidth" in data:
            phy.setBandwidth(data["bandwidth"],data["x"])
        if "buffer_size" in data:
            phy.setBufferSize(data["buffer_size"],data["x"])
        callback = {"contents": "done"}
        return jsonify(callback), 200