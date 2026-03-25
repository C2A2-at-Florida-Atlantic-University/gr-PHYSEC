from gnuradio import gr
import numpy as np, pmt

class blk(gr.sync_block):
    def __init__(self, num_items=8192):
        gr.sync_block.__init__(self, name='RX Gate', in_sig=[np.complex64], out_sig=[np.complex64])
        self.num_items = int(num_items); self.enabled = False; self.count = 0
        self.message_port_register_in(pmt.intern('ctrl'))
        self.message_port_register_out(pmt.intern('evt'))
        self.set_msg_handler(pmt.intern('ctrl'), self._on_ctrl)
    def _on_ctrl(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if val == 'start': self.enabled = True; self.count = 0
    def work(self, ins, outs):
        if not self.enabled: return 0
        n = min(len(ins[0]), len(outs[0]), self.num_items - self.count)
        outs[0][:n] = ins[0][:n]; self.count += n
        if self.count >= self.num_items:
            self.enabled = False
            self.message_port_pub(pmt.intern('evt'), pmt.intern('rx_collected'))
        return n
