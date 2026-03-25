from gnuradio import gr
import pmt, os, threading, time

class blk(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(self, name='Bob Controller', in_sig=None, out_sig=None)
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
    def handle_msg(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if val == 'start' and self.state == 'idle':
            # Ask Alice to prepare RX for Bob's TX probe
            self.message_port_pub(pmt.intern('peer'), pmt.intern('probe_req'))
            # Optional delay to allow Alice RX to arm
            try:
                tx_start_delay_ms = float(os.getenv('PHYSEC_TX_START_DELAY_MS', '0'))
            except Exception:
                tx_start_delay_ms = 0.0
            if tx_start_delay_ms > 0:
                time.sleep(tx_start_delay_ms / 1000.0)
            # Enable Bob TX and announce
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(False))
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_tx_start'))
            self.state = 'txing'
        elif val == 'alice_tx_start' and self.state in ('txing','tx_done'):
            # Mute Bob TX immediately, then start RX after optional delay
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(True))
            try:
                rx_delay_ms = float(os.getenv('PHYSEC_RX_DELAY_MS', '10'))
            except Exception:
                rx_delay_ms = 10.0
            if rx_delay_ms > 0:
                time.sleep(rx_delay_ms / 1000.0)
            self.message_port_pub(pmt.intern('rx_ctrl'), pmt.intern('start'))
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_collecting'))
            self.state = 'collecting'
        elif val == 'parity_recv':
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_parity_recv'))
        elif val in ('reconcile_ok','reconcile_fail'):
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_'+val))
            self.message_port_pub(pmt.intern('peer'), pmt.intern(val))
            self.state = 'idle'
        else:
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_evt_'+val))
    def handle_evt(self, msg):
        val = pmt.symbol_to_string(msg) if pmt.is_symbol(msg) else str(pmt.to_python(msg))
        if val == 'rx_collected' and self.state == 'collecting':
            self.message_port_pub(pmt.intern('tx_mute'), pmt.to_pmt(True))
            self.message_port_pub(pmt.intern('peer'), pmt.intern('bob_collect_done'))
            self.message_port_pub(pmt.intern('out'), pmt.intern('bob_collect_done'))
            self.state = 'processing'
