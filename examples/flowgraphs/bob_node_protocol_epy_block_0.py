from gnuradio import gr
import pmt

class blk(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self, name='Bob Controller', in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern('msg_in'))
        self.message_port_register_out(pmt.intern('tx_mute'))
        self.message_port_register_out(pmt.intern('out'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_ctrl)

    def handle_ctrl(self, msg):
        try:
            val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        except Exception:
            val = 'unknown'
        if 'start_tx' in val:
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(False))
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_start_tx'))
        elif 'stop_tx' in val:
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(True))
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_stop_tx'))
        elif 'collect' in val:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_collecting'))
        elif 'parity_recv' in val:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_parity_recv'))
        elif 'reconcile_ok' in val:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_reconcile_ok'))
        elif 'reconcile_fail' in val:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_reconcile_fail'))
        elif 'privacy_done' in val:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_privacy_done'))
        else:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_unknown'))
