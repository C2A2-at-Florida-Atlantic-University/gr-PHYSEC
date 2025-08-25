#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Bob Protocol Node - PHYSEC
# Author: Bob Node
# Description: Bob full-node skeleton in YAML GRC format (GR 3.10)
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import PHYSEC
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import zeromq
import bob_node_protocol_epy_block_0 as epy_block_0  # embedded python block



from gnuradio import qtgui

class bob_node_protocol(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Bob Protocol Node - PHYSEC", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Bob Protocol Node - PHYSEC")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "bob_node_protocol")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.vector_size = vector_size = 8192
        self.sdr_uri = sdr_uri = "ip:192.168.2.1"
        self.sample_rate = sample_rate = 2000000
        self.peer_ctrl_addr = peer_ctrl_addr = "tcp://127.0.0.1:9001"
        self.parity_sub_addr = parity_sub_addr = "tcp://127.0.0.1:9200"
        self.parity_len = parity_len = 127
        self.monitor_addr = monitor_addr = "tcp://127.0.0.1:9101"

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_sub_msg_source_parity = zeromq.sub_msg_source(parity_sub_addr, 100, False)
        self.zeromq_sub_msg_source_0 = zeromq.sub_msg_source(peer_ctrl_addr, 100, False)
        self.zeromq_pub_msg_sink_0 = zeromq.pub_msg_sink(monitor_addr, 100, False)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.epy_block_0 = epy_block_0.blk()
        self.blocks_stream_to_vector_1 = blocks.stream_to_vector(gr.sizeof_char*1, parity_len)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vector_size)
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, vector_size)
        self.blocks_file_sink_vec = blocks.file_sink(gr.sizeof_gr_complex*8192, '', False)
        self.blocks_file_sink_vec.set_unbuffered(False)
        self.blocks_file_sink_key_0 = blocks.file_sink(gr.sizeof_char*128, '', False)
        self.blocks_file_sink_key_0.set_unbuffered(False)
        self.analog_sig_source_x_0 = analog.sig_source_c(sample_rate, analog.GR_SIN_WAVE, 1000000, 1, 0, 0)
        self.PHYSEC_spectrogram_block_0 = PHYSEC.spectrogram_block(512, 8192)
        self.PHYSEC_reconciliation_block_0 = PHYSEC.reconciliation_block(255, 128, 512)
        self.PHYSEC_privacy_amplification_block_0 = PHYSEC.privacy_amplification_block('sha3_512')
        self.PHYSEC_feature_quantization_block_0 = PHYSEC.feature_quantization_block('mean_threshold')
        self.PHYSEC_feature_extraction_block_0 = PHYSEC.feature_extraction_block('/path/to/model.onnx')


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_0, 'out'), (self.zeromq_pub_msg_sink_0, 'in'))
        self.msg_connect((self.zeromq_sub_msg_source_0, 'out'), (self.epy_block_0, 'msg_in'))
        self.msg_connect((self.zeromq_sub_msg_source_parity, 'out'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.PHYSEC_feature_extraction_block_0, 0), (self.PHYSEC_feature_quantization_block_0, 0))
        self.connect((self.PHYSEC_feature_quantization_block_0, 0), (self.PHYSEC_reconciliation_block_0, 0))
        self.connect((self.PHYSEC_privacy_amplification_block_0, 0), (self.blocks_file_sink_key_0, 0))
        self.connect((self.PHYSEC_reconciliation_block_0, 0), (self.PHYSEC_privacy_amplification_block_0, 0))
        self.connect((self.PHYSEC_spectrogram_block_0, 0), (self.PHYSEC_feature_extraction_block_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.PHYSEC_spectrogram_block_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_file_sink_vec, 0))
        self.connect((self.blocks_stream_to_vector_1, 0), (self.PHYSEC_reconciliation_block_0, 1))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_stream_to_vector_1, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "bob_node_protocol")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_vector_size(self):
        return self.vector_size

    def set_vector_size(self, vector_size):
        self.vector_size = vector_size
        self.blocks_head_0.set_length(self.vector_size)

    def get_sdr_uri(self):
        return self.sdr_uri

    def set_sdr_uri(self, sdr_uri):
        self.sdr_uri = sdr_uri

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.sample_rate)

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




def main(top_block_cls=bob_node_protocol, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
