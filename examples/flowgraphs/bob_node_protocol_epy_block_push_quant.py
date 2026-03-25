from gnuradio import gr
import numpy as np, socket, json, time, threading, os

class blk(gr.sync_block):
    def __init__(self, vlen=512, node_name="Bob"):
        gr.sync_block.__init__(self, name='Monitor Push Quantized', in_sig=[(np.uint8, int(vlen))], out_sig=None)
        self.vlen = int(vlen)
        self.node_name = node_name
        self.host = os.getenv('MONITOR_HOST', '192.168.0.142')
        self.port = int(os.getenv('MONITOR_PORT', '9999'))
        self.sock = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            s.settimeout(None)
            self.sock = s
        except Exception:
            self.sock = None

    def _send(self, payload):
        try:
            if self.sock is None:
                self._connect()
            if self.sock:
                data = (json.dumps(payload) + "\n").encode('utf-8')
                with self._lock:
                    self.sock.sendall(data)
        except Exception:
            self.sock = None

    def work(self, input_items, output_items):
        vecs = input_items[0]
        for v in vecs:
            try:
                data_list = [int(x) for x in (v.tolist() if hasattr(v, 'tolist') else list(v))]
                payload = {"type": "data_push", "data_type": "quantized_bits", "node_name": self.node_name, "data": data_list, "timestamp": time.time()}
                self._send(payload)
            except Exception:
                pass
        return len(vecs)
