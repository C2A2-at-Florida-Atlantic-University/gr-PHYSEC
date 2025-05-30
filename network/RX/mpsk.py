#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: PSK_RX
# Author: Jose Sanchez
# GNU Radio version: 3.10.6.0

from gnuradio import digital, gr, blocks, uhd
from gnuradio.filter import firdes
import math

class MPSK(gr.top_block):
        
    def __init__(self,
                samp_rate=1000000,
                sps = 4,
                gain=10,
                freq=2400000000,
                buffer_size=32768,
                bandwidth=20000000,
                SDR_ADDR="",
                UDP_port=40860,
                M=2):

        gr.top_block.__init__(self, "PSK_RX")

        ##################################################
        # Variables
        ##################################################
        self.sps = sps
        self.nfilts = nfilts = int(32*2)
        self.samp_rate = samp_rate
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 11*sps*nfilts)
        self.phase_bw = phase_bw = (math.pi*2)/100
        self.gain = gain
        self.freq = freq
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.SDR_ADDR = SDR_ADDR
        self.UDP_port = UDP_port
        self.M = M

        ##################################################
        # Blocks
        ##################################################

        self.udp_sink = blocks.udp_sink(
            gr.sizeof_gr_complex, "127.0.0.1", self.UDP_port,
            self.buffer_size, True
        )
        
        # self.iio_pluto_source_0=iio.pluto_source(self.SDR_ID, self.freq, self.samp_rate, self.bandwidth, 
        #                                         self.buffer_size, True, True, True, 'manual', self.gain, '', True)
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
        self.digital_symbol_sync_xx_0_0 = digital.symbol_sync_cc(
            digital.TED_SIGNAL_TIMES_SLOPE_ML,
            sps,
            phase_bw,
            1.0,
            1.0,
            1.5,
            2,
            digital.constellation_bpsk().base(),
            digital.IR_PFB_MF,
            nfilts,
            rrc_taps)
        self.digital_costas_loop_cc_0_0 = digital.costas_loop_cc(phase_bw, M, False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.digital_costas_loop_cc_0_0, 0), (self.blocks_udp_sink_1, 0))
        self.connect((self.digital_symbol_sync_xx_0_0, 0), (self.digital_costas_loop_cc_0_0, 0))
        self.connect((self.usrp_source, 0), (self.digital_symbol_sync_xx_0_0, 0))


    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_source.set_samp_rate(self.samp_rate)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_costas_loop_cc_0_0.set_loop_bandwidth(self.phase_bw)
        self.digital_symbol_sync_xx_0_0.set_loop_bandwidth(self.phase_bw)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_source.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.usrp_source.set_center_freq(self.freq, 0)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        # self.iio_pluto_source_0.set_params(self.freq, self.samp_rate, self.bandwidth, True, True, True, 'manual', self.gain, '', True)

    def get_M(self):
        return self.M

    def set_M(self, M):
        self.M = M

def main(top_block_cls=MPSK):
    tb = top_block_cls()

    tb.start()

    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
