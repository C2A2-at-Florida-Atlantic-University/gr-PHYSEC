#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: PSK_TX
# Author: Jose Sanchez
# GNU Radio version: 3.10.6.0

from gnuradio import blocks, digital, gr, uhd
import numpy
import math

class MPSK(gr.top_block):

    def __init__(self,
                samp_rate=1000000,
                sps=4,
                gain=0,
                freq=2400000000,
                buffer_size=32768,
                bandwidth=20000000,
                SDR_ADDR="",
                M=2):
        gr.top_block.__init__(self, "MPSK")

        ##################################################
        # Variables
        ##################################################
        self.M = M
        self.sps = sps
        self.samp_rate = samp_rate
        self.gain = gain
        self.freq = freq
        self.excess_bw = excess_bw = 0.35
        self.buffer_size = buffer_size
        self.bitsPerSymbol = bps = int(math.log(M,2))
        self.bandwidth = bandwidth
        self.SDR_ADDR = SDR_ADDR
        if M == 2:
            self.ConstObj = digital.constellation_bpsk().base()
        elif M == 4:
            self.ConstObj = digital.constellation_qpsk().base()
        elif M == 8:
            self.ConstObj = digital.constellation_8psk().base()        
            
        ##################################################
        # Blocks
        ##################################################

        # self.iio_pluto_sink_0=iio.pluto_sink(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, gain, '', True)
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
        # choose TX port on B200-series / X300-series
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_max_output_buffer(self.max_buf)
        self.digital_constellation_modulator_0=digital.generic_mod(
            constellation=self.ConstObj,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(1, bps, "", False, gr.GR_LSB_FIRST)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 2, 1000))), True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.usrp_sink, 0))


    def get_M(self):
        return self.M

    def set_M(self, M):
        self.M = M
        self.set_bps(int(math.log(self.M,2)))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_sink.set_samp_rate(self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_sink.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.usrp_sink.set_center_freq(self.freq, 0)

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def get_bps(self):
        return self.bps

    def set_bps(self, bps):
        self.bps = bps
        self.blocks_repack_bits_bb_0_0.set_k_and_l(1,self.bps)

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        # self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_SDR_ID(self):
        return self.SDR_ID

    def set_SDR_ID(self, SDR_ID):
        self.SDR_ID = SDR_ID

    def get_ConstObj(self):
        return self.ConstObj

    def set_ConstObj(self, ConstObj):
        self.ConstObj = ConstObj

def main(top_block_cls=MPSK):
    tb = top_block_cls()

    tb.start()
    
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
