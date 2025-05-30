#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Rx
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

from gnuradio import gr, blocks, uhd
import sys
import time

class Sinusoid(gr.top_block):

    def __init__(self,
                samp_rate=1000000,
                gain=31,
                freq=3550000000,
                buffer_size=8192,
                bandwidth=20000000,
                SDR_ADDR="",
                UDP_port=40860):
        gr.top_block.__init__(self, "Rx")
        
        ##################################################
        # Variables
        ##################################################
        self.samp_rate=samp_rate
        self.gain=gain
        self.freq=freq
        self.buffer_size=buffer_size
        self.SDR_ADDR=SDR_ADDR
        self.UDP_port=UDP_port
        self.bandwidth=bandwidth

        ##################################################
        # Blocks
        ##################################################
        self.udp_sink = blocks.udp_sink(
            gr.sizeof_gr_complex*1, 
            '127.0.0.1', 
            self.UDP_port, 
            self.buffer_size, 
            True
        )
        # self.iio_pluto_source_0=iio.pluto_source(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, True, True, 'manual', gain, '', True)
        # UHD USRP source block
        self.max_buf = 1024*1024  
        self.usrp_source = uhd.usrp_source(
            ",".join((self.SDR_ADDR, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            )
        )
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.usrp_source.set_center_freq(self.freq, 0)
        self.usrp_source.set_gain(self.gain, 0)
        self.usrp_source.set_antenna("RX2", 0)
        self.usrp_source.set_max_output_buffer(self.max_buf)
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0x10, 0x10, 0)
        ##################################################
        # Connections
        ##################################################
        self.connect((self.usrp_source, 0), (self.udp_sink, 0))

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.usrp_source.set_samp_rate(self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain=gain
        self.usrp_source.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq=freq
        self.usrp_source.set_center_freq(self.freq, 0)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size=buffer_size
    
    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth=bandwidth
        # self.iio_pluto_source_0.set_params(self.freq, self.samp_rate, self.bandwidth, True, True, True, 'manual', self.gain, '', True)
    
    def start(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0x10, 0x10, 0)
        super().start()

    def stop(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0xFFFFFFFF, 0x0, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0xFFFFFFFF, 0x0, 0)
        return super().stop()

def main(top_block_cls=Sinusoid):
    
    tb = top_block_cls()

    tb.start()

    time.sleep(10)

    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
