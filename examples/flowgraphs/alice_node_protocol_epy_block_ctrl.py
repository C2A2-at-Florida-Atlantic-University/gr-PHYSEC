from gnuradio import gr
import pmt, os, threading, time

class blk(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self, name='Alice Controller', in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern('msg_in'))
        self.message_port_register_in(pmt.intern('evt_in'))
        self.message_port_register_out(pmt.intern('tx_mute'))
        self.message_port_register_out(pmt.intern('out'))
        self.message_port_register_out(pmt.intern('peer'))
        self.message_port_register_out(pmt.intern('rx_ctrl'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_msg)
        self.set_msg_handler(pmt.intern('evt_in'), self.handle_evt)
        self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(True))
        self.state = 'idle'
    def _start_rx(self):
        self.message_port_pub(pmt.intern('rx_ctrl'), pmt.intern('start'))
    def handle_msg(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if val == 'probe_req' and self.state == 'idle':
            self.message_port_pub(pmt.intern('out'), pmt.intern('alice_accept'))
            time.sleep(0.001)
            self.message_port_pub(pmt.intern('rx_ctrl'), pmt.intern('start'))
            self.state = 'collecting'
        elif val == 'bob_collect_done' and self.state == 'txing':
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(True))
            self.message_port_pub(pmt.intern('out'), pmt.intern('alice_tx_stop'))
            self.state = 'processing'
        elif val in ('reconcile_ok','reconcile_fail'):
            self.message_port_pub(pmt.intern('out'), pmt.intern('alice_'+val))
            self.state = 'idle'
        else:
            self.message_port_pub(pmt.intern('out'), pmt.intern('alice_evt_'+val))
    def handle_evt(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if val == 'rx_collected' and self.state == 'collecting':
            # Optional delay before switching to TX to let Bob prepare RX
            try:
                rx2tx_delay_ms = float(os.getenv('PHYSEC_RX2TX_DELAY_MS', '10'))
            except Exception:
                rx2tx_delay_ms = 10.0
            if rx2tx_delay_ms > 0:
                time.sleep(rx2tx_delay_ms / 1000.0)
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(False))
            self.message_port_pub(pmt.intern('peer'), pmt.intern('alice_tx_start'))
            self.message_port_pub(pmt.intern('out'), pmt.intern('alice_tx_start'))
            self.state = 'txing'
