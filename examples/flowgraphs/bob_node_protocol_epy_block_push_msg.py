from gnuradio import gr
import pmt, socket, json, time, threading, os

class blk(gr.basic_block):
    def __init__(self, node_name="Bob", count_run_on="bob_tx_start"):
        gr.basic_block.__init__(self, name='Monitor Push (msg)', in_sig=None, out_sig=None)
        self.node_name = node_name
        self.count_run_on = str(count_run_on) if count_run_on else ""
        self.host = os.getenv('MONITOR_HOST', '192.168.0.142')
        self.port = int(os.getenv('MONITOR_PORT', '9999'))
        self.run_number = 0
        self.sock = None
        self._lock = threading.Lock()
        self._connect()
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle_msg)

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

    def handle_msg(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if self.count_run_on and val == self.count_run_on:
            self.run_number += 1
            self._send({"type": "data_push", "data_type": "run_update", "node_name": self.node_name, "data": {"run_number": self.run_number, "action": "start"}, "timestamp": time.time()})
        self._send({"type": "data_push", "data_type": "protocol_step", "node_name": self.node_name, "data": {"step": val}, "timestamp": time.time()})
