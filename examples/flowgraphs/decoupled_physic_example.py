#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: PHYSEC Decoupled Blocks Example
# Author: gr-PHYSEC Decoupled Example
# Description: Demonstrates the decoupled PHYSEC framework
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

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import PHYSEC
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio



from gnuradio import qtgui

class decoupled_physic_example(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "PHYSEC Decoupled Blocks Example", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("PHYSEC Decoupled Blocks Example")
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

        self.settings = Qt.QSettings("GNU Radio", "decoupled_physic_example")

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
        self.modelPath = modelPath = "/workspace/data/gr-PHYSEC/models/QExtractor.onnx"
        self.fftWindow = fftWindow = 512
        self.centerFrequency = centerFrequency = 2400000000

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            sampleRate, #bw
            'QT GUI Freq Sink', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['IQ Spectrum', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
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
        self.iio_pluto_sink_0 = iio.fmcomms2_sink_fc32("ip:192.168.65.254" if "ip:192.168.65.254" else iio.get_pluto_uri(), [True, True], 32768, False)
        self.iio_pluto_sink_0.set_len_tag_key('')
        self.iio_pluto_sink_0.set_bandwidth(20000000)
        self.iio_pluto_sink_0.set_frequency(2400000000)
        self.iio_pluto_sink_0.set_samplerate(sampleRate)
        self.iio_pluto_sink_0.set_attenuation(0, 10.0)
        self.iio_pluto_sink_0.set_filter_params('Auto', '', 0, 0)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, sampleRate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vectorSize)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_char*128, '/tmp/keys.txt', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*512, '/tmp/physic_quantized_features.txt', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_sig_source_x_0 = analog.sig_source_c(sampleRate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        self.PHYSEC_spectrogram_block_0 = PHYSEC.spectrogram_block(fftWindow, vectorSize)
        self.PHYSEC_reconciliation_block_0 = PHYSEC.reconciliation_block(255, 128, 512)
        self.PHYSEC_privacy_amplification_block_0 = PHYSEC.privacy_amplification_block('sha3_512')
        self.PHYSEC_parity_generation_block_0 = PHYSEC.parity_generation_block(255, 128, 512)
        self.PHYSEC_feature_quantization_block_0 = PHYSEC.feature_quantization_block('mean_threshold')
        self.PHYSEC_feature_extraction_block_0 = PHYSEC.feature_extraction_block(modelPath)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.PHYSEC_feature_extraction_block_0, 0), (self.PHYSEC_feature_quantization_block_0, 0))
        self.connect((self.PHYSEC_feature_quantization_block_0, 0), (self.PHYSEC_parity_generation_block_0, 0))
        self.connect((self.PHYSEC_feature_quantization_block_0, 0), (self.PHYSEC_reconciliation_block_0, 0))
        self.connect((self.PHYSEC_feature_quantization_block_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.PHYSEC_parity_generation_block_0, 0), (self.PHYSEC_reconciliation_block_0, 1))
        self.connect((self.PHYSEC_privacy_amplification_block_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.PHYSEC_reconciliation_block_0, 0), (self.PHYSEC_privacy_amplification_block_0, 0))
        self.connect((self.PHYSEC_spectrogram_block_0, 0), (self.PHYSEC_feature_extraction_block_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.iio_pluto_sink_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.PHYSEC_spectrogram_block_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.qtgui_freq_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "decoupled_physic_example")
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
        self.analog_sig_source_x_0.set_sampling_freq(self.sampleRate)
        self.blocks_throttle_0.set_sample_rate(self.sampleRate)
        self.iio_pluto_sink_0.set_samplerate(self.sampleRate)
        self.iio_pluto_source_0.set_samplerate(self.sampleRate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.sampleRate)

    def get_modelPath(self):
        return self.modelPath

    def set_modelPath(self, modelPath):
        self.modelPath = modelPath

    def get_fftWindow(self):
        return self.fftWindow

    def set_fftWindow(self, fftWindow):
        self.fftWindow = fftWindow

    def get_centerFrequency(self):
        return self.centerFrequency

    def set_centerFrequency(self, centerFrequency):
        self.centerFrequency = centerFrequency
        self.iio_pluto_source_0.set_frequency(self.centerFrequency)




def main(top_block_cls=decoupled_physic_example, options=None):

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
