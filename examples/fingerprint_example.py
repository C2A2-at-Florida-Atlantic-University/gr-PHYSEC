#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: PHYSEC Fingerprint Example - FFT Spectrogram
# Author: gr-PHYSEC Example
# Description: Demonstrates channel fingerprinting using the PHYSEC block with FFT spectrogram generation
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
from gnuradio import iio



from gnuradio import qtgui

class fingerprint_example(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "PHYSEC Fingerprint Example - FFT Spectrogram", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("PHYSEC Fingerprint Example - FFT Spectrogram")
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

        self.settings = Qt.QSettings("GNU Radio", "fingerprint_example")

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
        self.vectorSize = vectorSize = 8192
        self.sampleRate = sampleRate = 1000000
        self.keyLength = keyLength = 128
        self.centerFrequency = centerFrequency = 2400000000

        ##################################################
        # Blocks
        ##################################################
        self.iio_pluto_source_0 = iio.fmcomms2_source_fc32("ip:192.168.65.254" if "ip:192.168.65.254" else iio.get_pluto_uri(), [True, True], 32768)
        self.iio_pluto_source_0.set_len_tag_key('packet_len')
        self.iio_pluto_source_0.set_frequency(centerFrequency)
        self.iio_pluto_source_0.set_samplerate(sampleRate)
        self.iio_pluto_source_0.set_gain_mode(0, 'slow_attack')
        self.iio_pluto_source_0.set_gain(0, 64)
        self.iio_pluto_source_0.set_quadrature(True)
        self.iio_pluto_source_0.set_rfdc(True)
        self.iio_pluto_source_0.set_bbdc(True)
        self.iio_pluto_source_0.set_filter_params('Auto', '', 0, 0)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, sampleRate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vectorSize)
        self.blocks_message_debug_1 = blocks.message_debug(True)
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.PHYSEC_fingerprint_block_0 = PHYSEC.fingerprint_block('/workspace/gr-PHYSEC/models/QExtractor.onnx', 'quadruplet', vectorSize, sampleRate, centerFrequency, keyLength)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_stream_to_vector_0, 0), (self.PHYSEC_fingerprint_block_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.blocks_throttle_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "fingerprint_example")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_vectorSize(self):
        return self.vectorSize

    def set_vectorSize(self, vectorSize):
        self.vectorSize = vectorSize

    def get_sampleRate(self):
        return self.sampleRate

    def set_sampleRate(self, sampleRate):
        self.sampleRate = sampleRate
        self.blocks_throttle_0.set_sample_rate(self.sampleRate)
        self.iio_pluto_source_0.set_samplerate(self.sampleRate)

    def get_keyLength(self):
        return self.keyLength

    def set_keyLength(self, keyLength):
        self.keyLength = keyLength

    def get_centerFrequency(self):
        return self.centerFrequency

    def set_centerFrequency(self, centerFrequency):
        self.centerFrequency = centerFrequency
        self.iio_pluto_source_0.set_frequency(self.centerFrequency)




def main(top_block_cls=fingerprint_example, options=None):

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
