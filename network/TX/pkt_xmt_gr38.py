#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: pkt_xmt_gr38
# Author: Barry Duggan
# Description: packet transmit (for GNURadio 3.8)
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

import time
from gnuradio import blocks, digital, filter, gr, uhd
# from gnuradio.pdu import pdu_to_tagged_stream

import pmt
import sys
# from packet_format_gr38 import packet_format
sys.path.append('../')
from TX.packet_format_gr38 import packet_format
# import previous path ../ using sys


class packetTransmit(gr.top_block):
    def __init__(self,
                input,
                input_len,
                samp_rate=600e3,
                sps = 2,
                gain=10,
                freq=3.55e9,
                buffer_size=32768,
                bandwidth=20000000,
                SDR_ADDR=""):
        
        gr.top_block.__init__(self, "txData")
        ##################################################
        # Variables
        ##################################################
        self.usrp_rate=usrp_rate=768000
        self.sps=sps
        self.samp_rate=samp_rate
        self.rs_ratio=rs_ratio=1.040
        self.gain=gain
        self.freq=freq
        self.excess_bw=excess_bw=0.35
        self.buffer_size=buffer_size
        self.bpsk=bpsk=digital.constellation_bpsk().base()
        self.bandwidth=bandwidth
        self.SDR_ADDR=SDR_ADDR
        self.input=input
        self.input_len=input_len
        self.max_buf = 1024*1024  
        ##################################################
        # Blocks
        ##################################################
        self.packet_format_gr38=packet_format()
        self.mmse_resampler_xx_0=filter.mmse_resampler_cc(0, 1.0/((usrp_rate/samp_rate)*rs_ratio))
        # self.iio_pluto_sink_0=iio.pluto_sink(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, gain, '', True)
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
        # choose TX port on B200-series / X300-series
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_clock_rate(30.72e6, uhd.ALL_MBOARDS)
        self.usrp_sink.set_max_output_buffer(self.max_buf)
        self.usrp_sink.set_time_unknown_pps(uhd.time_spec())
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
        
        self.digital_crc32_async_bb_1=digital.crc32_async_bb(False)
        self.digital_constellation_modulator_0=digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False)
        self.blocks_throttle_0=blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_pdu_to_tagged_stream_0=blocks.pdu_to_tagged_stream(blocks.byte_t, 'packet_len')
        # self.blocks_pdu_to_tagged_stream_0 = pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.blocks_multiply_const_vxx_0=blocks.multiply_const_cc(0.5)
        self.blocks_message_strobe_0=blocks.message_strobe(pmt.cons(pmt.PMT_NIL,pmt.init_u8vector(input_len ,input)), 100)
        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.digital_crc32_async_bb_1, 'in'))
        self.msg_connect((self.digital_crc32_async_bb_1, 'out'), (self.packet_format_gr38, 'PDU_in'))
        self.msg_connect((self.packet_format_gr38, 'PDU_out0'), (self.blocks_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.mmse_resampler_xx_0, 0))
        self.connect((self.blocks_pdu_to_tagged_stream_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.usrp_sink, 0))
    
    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate=usrp_rate
        self.mmse_resampler_xx_0.set_resamp_ratio(1.0/((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_sps(self):
        return self.sps

    def set_data(self,input_len,input):
        self.input = input
        self.input_len = input_len
        self.blocks_message_strobe_0=blocks.message_strobe(pmt.cons(pmt.PMT_NIL,pmt.init_u8vector(input_len ,input)), 100)

    def set_sps(self, sps):
        self.sps=sps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.mmse_resampler_xx_0.set_resamp_ratio(1.0/((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_rs_ratio(self):
        return self.rs_ratio

    def set_rs_ratio(self, rs_ratio):
        self.rs_ratio=rs_ratio
        self.mmse_resampler_xx_0.set_resamp_ratio(1.0/((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain=gain
        self.usrp_sink.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq=freq
        self.usrp_sink.set_center_freq(self.freq, 0)

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw=excess_bw

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size=buffer_size

    def get_bpsk(self):
        return self.bpsk

    def set_bpsk(self, bpsk):
        self.bpsk=bpsk

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

def str_to_length_and_decimals(text):
    if isinstance(text, bytes):
        values=[c for c in text]
    else:
        values=[ord(c) for c in text]
    length=len(values)
    return length, values
    
def main(top_block_cls=packetTransmit):

    input = b'Hello World'
    length, values = str_to_length_and_decimals(input)
    tb = top_block_cls(
        input=values,
        input_len=length,
        samp_rate=600e3,
        sps=2,
        gain=10,
        freq=3.450e9,
        buffer_size=32768,
        bandwidth=20000000,
        SDR_ADDR="",
    )

    tb.start()
    time.sleep(20)
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
