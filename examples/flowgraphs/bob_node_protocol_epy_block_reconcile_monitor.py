from gnuradio import gr
import pmt
import numpy as np

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self, name='Reconcile Monitor', in_sig=[(np.uint8,1)], out_sig=None)
        self.message_port_register_out(pmt.intern('out'))
    def work(self, input_items, output_items):
        flags = input_items[0]
        for f in flags:
            val = int(f[0]) if hasattr(f, '__len__') else int(f)
            msg = pmt.intern('reconcile_ok') if val != 0 else pmt.intern('reconcile_fail')
            self.message_port_pub(pmt.intern('out'), msg)
        return len(flags)
