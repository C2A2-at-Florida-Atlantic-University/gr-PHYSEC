#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Tx
# Author: root
# GNU Radio version: v3.8.5.0-6-g57bd109d

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import analog, gr, uhd
import sys
import time

class Sinusoid(gr.top_block):

    def __init__(self,
                samp_rate=1000000,
                gain=31,
                freq=3.55e9,
                buffer_size=8192,
                bandwidth=20000000,
                SDR_ADDR=""):
        gr.top_block.__init__(self, "Sinusoid")
        ##################################################
        # Variables
        ##################################################
        self.samp_rate=samp_rate 
        self.gain=gain 
        self.freq=freq 
        self.buffer_size=buffer_size 
        self.bandwidth=bandwidth 
        self.SDR_ADDR=SDR_ADDR 

        ##################################################
        # Blocks
        ##################################################
        # self.iio_pluto_sink_0 = iio.pluto_sink(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, gain, '', True)
        self.max_buf = 1024*1024  
        self.usrp_sink = uhd.usrp_sink(
            # device address string: blank => first USRP found
            ",".join((self.SDR_ADDR, "")),
            # stream args: one channel of complex floats
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            ),
            ""  # XML or args string (unused here)
        )
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.usrp_sink.set_center_freq(self.freq, 0)
        self.usrp_sink.set_gain(self.gain, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_clock_rate(30.72e6, uhd.ALL_MBOARDS)
        self.usrp_sink.set_max_output_buffer(self.max_buf)
        self.usrp_sink.set_time_unknown_pps(uhd.time_spec())
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
        self.analog_sig_source_x_0=analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)
        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.usrp_sink, 0))

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.usrp_sink.set_samp_rate(self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain=gain
        self.usrp_sink.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq=freq
        self.analog_sig_source_x_0.set_frequency(self.freq)
        self.usrp_sink.set_center_freq(self.freq, 0)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size=buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth=bandwidth
        # self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_SDR_ID(self):
        return self.SDR_ID

    def set_SDR_ID(self, SDR_ID):
        self.SDR_ID=SDR_ID
        
    def start(self):
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
        super().start()
        
    def stop(self):
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0xFFFFFFFF, 0x0, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0xFFFFFFFF, 0x0, 0)
        return super().stop()

def main(top_block_cls=Sinusoid, options=None):
    tb = top_block_cls()
    tb.start()
    time.sleep(20)
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
