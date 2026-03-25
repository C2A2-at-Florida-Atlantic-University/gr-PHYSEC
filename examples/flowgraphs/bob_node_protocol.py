#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Bob Protocol Node - PHYSEC
# Author: Bob Node
# Description: Bob full-node skeleton in YAML GRC format (GR 3.10)
# GNU Radio version: 3.10.12.0

from gnuradio import PHYSEC
from gnuradio import analog
from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import network
from gnuradio import zeromq
import bob_node_protocol_epy_block_0 as epy_block_0  # embedded python block
import bob_node_protocol_epy_block_push_iq as epy_block_push_iq  # embedded python block
import bob_node_protocol_epy_block_push_msg as epy_block_push_msg  # embedded python block
import bob_node_protocol_epy_block_push_spec as epy_block_push_spec  # embedded python block
import bob_node_protocol_epy_block_rx_gate as epy_block_rx_gate  # embedded python block
import threading




class bob_node_protocol(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Bob Protocol Node - PHYSEC", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.vector_size = vector_size = 8192
        self.sdr_uri = sdr_uri = "ip:192.168.2.1"
        self.sample_rate = sample_rate = 1000000
        self.peer_trigger_addr = peer_trigger_addr = "tcp://192.168.0.142:9103"
        self.peer_ctrl_pub_addr = peer_ctrl_pub_addr = "tcp://127.0.0.1:9102"
        self.peer_ctrl_addr = peer_ctrl_addr = "tcp://127.0.0.1:9101"
        self.parity_sub_addr = parity_sub_addr = "tcp://127.0.0.1:9200"
        self.parity_len = parity_len = 127
        self.monitor_addr = monitor_addr = "tcp://127.0.0.1:9101"
        self.freq = freq = 2400000000

        ##################################################
        # Blocks
        ##################################################

        self.zeromq_sub_msg_source_trigger = zeromq.sub_msg_source(peer_trigger_addr, 100, False)
        self.zeromq_sub_msg_source_0 = zeromq.sub_msg_source(peer_ctrl_addr, 100, False)
        self.zeromq_pub_msg_sink_ctrl = zeromq.pub_msg_sink(peer_ctrl_pub_addr, 100, True)
        self.zeromq_pub_msg_sink_0 = zeromq.pub_msg_sink(monitor_addr, 100, False)
        self.network_udp_source_0 = network.udp_source(gr.sizeof_gr_complex, 1, 3000, 0, 1472, False, False, False)
        self.network_udp_sink_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2000, 0, 1472, False)
        self.epy_block_rx_gate = epy_block_rx_gate.blk(num_items=vector_size)
        self.epy_block_push_spec = epy_block_push_spec.blk(node_name="Bob")
        self.epy_block_push_msg = epy_block_push_msg.blk(node_name="Bob", count_run_on="bob_tx_start")
        self.epy_block_push_iq = epy_block_push_iq.blk(vlen=vector_size, node_name="Bob")
        self.epy_block_0 = epy_block_0.blk()
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vector_size)
        self.blocks_mute_xx_0 = blocks.mute_cc(bool(True))
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(sample_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.05, 0)
        self.PHYSEC_spectrogram_block_0 = PHYSEC.spectrogram_block(512, 8192)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_0, 'rx_ctrl'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.epy_block_0, 'tx_mute'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.epy_block_0, 'peer'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.epy_block_0, 'out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.epy_block_0, 'tx_mute'), (self.blocks_mute_xx_0, 'set_mute'))
        self.msg_connect((self.epy_block_0, 'out'), (self.epy_block_push_msg, 'in'))
        self.msg_connect((self.epy_block_0, 'rx_ctrl'), (self.epy_block_rx_gate, 'ctrl'))
        self.msg_connect((self.epy_block_0, 'out'), (self.zeromq_pub_msg_sink_0, 'in'))
        self.msg_connect((self.epy_block_0, 'peer'), (self.zeromq_pub_msg_sink_ctrl, 'in'))
        self.msg_connect((self.epy_block_rx_gate, 'evt'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.epy_block_rx_gate, 'evt'), (self.epy_block_0, 'evt_in'))
        self.msg_connect((self.zeromq_sub_msg_source_0, 'out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.zeromq_sub_msg_source_0, 'out'), (self.epy_block_0, 'msg_in'))
        self.msg_connect((self.zeromq_sub_msg_source_trigger, 'out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.zeromq_sub_msg_source_trigger, 'out'), (self.epy_block_0, 'msg_in'))
        self.connect((self.PHYSEC_spectrogram_block_0, 0), (self.epy_block_push_spec, 0))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_mute_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.blocks_mute_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.PHYSEC_spectrogram_block_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.epy_block_push_iq, 0))
        self.connect((self.epy_block_rx_gate, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.network_udp_source_0, 0), (self.epy_block_rx_gate, 0))


    def get_vector_size(self):
        return self.vector_size

    def set_vector_size(self, vector_size):
        self.vector_size = vector_size
        self.epy_block_push_iq.vlen = self.vector_size
        self.epy_block_rx_gate.num_items = self.vector_size

    def get_sdr_uri(self):
        return self.sdr_uri

    def set_sdr_uri(self, sdr_uri):
        self.sdr_uri = sdr_uri

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.analog_sig_source_x_0_0.set_sampling_freq(self.sample_rate)

    def get_peer_trigger_addr(self):
        return self.peer_trigger_addr

    def set_peer_trigger_addr(self, peer_trigger_addr):
        self.peer_trigger_addr = peer_trigger_addr

    def get_peer_ctrl_pub_addr(self):
        return self.peer_ctrl_pub_addr

    def set_peer_ctrl_pub_addr(self, peer_ctrl_pub_addr):
        self.peer_ctrl_pub_addr = peer_ctrl_pub_addr

    def get_peer_ctrl_addr(self):
        return self.peer_ctrl_addr

    def set_peer_ctrl_addr(self, peer_ctrl_addr):
        self.peer_ctrl_addr = peer_ctrl_addr

    def get_parity_sub_addr(self):
        return self.parity_sub_addr

    def set_parity_sub_addr(self, parity_sub_addr):
        self.parity_sub_addr = parity_sub_addr

    def get_parity_len(self):
        return self.parity_len

    def set_parity_len(self, parity_len):
        self.parity_len = parity_len

    def get_monitor_addr(self):
        return self.monitor_addr

    def set_monitor_addr(self, monitor_addr):
        self.monitor_addr = monitor_addr

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq




def main(top_block_cls=bob_node_protocol, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
