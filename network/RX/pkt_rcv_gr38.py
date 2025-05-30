#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: pkt_rcv_gr38
# Author: Barry Duggan
# Description: packet receive (for GNURadio 3.8)
# GNU Radio version: v3.8.5.0-6-g57bd109d

import time
from gnuradio import blocks, digital, filter, gr, uhd

class packetReceive(gr.top_block):

    def __init__(self,
                samp_rate=600e3,
                sps = 2,
                gain=60,
                freq=3.550e9,
                buffer_size=32768,
                bandwidth=20000000,
                SDR_ADDR="",
                UDP_port=40860):
        gr.top_block.__init__(self, "packetReceive")

        ##################################################
        # Variables
        ##################################################
        self.usrp_rate=768000
        self.thresh=thresh=1
        self.sps=sps
        self.samp_rate=samp_rate
        self.rs_ratio=1.0
        self.phase_bw=0.0628
        self.order=2
        self.gain=gain
        self.freq=freq
        self.excess_bw=0.35
        self.buffer_size=buffer_size
        self.bandwidth=bandwidth
        self.SDR_ADDR    = SDR_ADDR
        self.UDP_port=UDP_port
        self.bpsk=digital.constellation_bpsk().base()

        ##################################################
        # Blocks
        ##################################################
        self.mmse_resampler_xx_0=filter.mmse_resampler_cc(
            0, 
            ((self.usrp_rate/self.samp_rate)*self.rs_ratio)
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
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0x10, 0x10, 0)
        
        self.digital_symbol_sync_xx_0=digital.symbol_sync_cc(
            digital.TED_MUELLER_AND_MULLER,
            self.sps,self.phase_bw,
            1.0,1.0,1.5,1,
            digital.constellation_bpsk().base(),
            digital.IR_MMSE_8TAP,128,[]
        )
        self.digital_map_bb_0=digital.map_bb([0,1])
        self.digital_lms_dd_equalizer_cc_0=digital.lms_dd_equalizer_cc(15, 0.000001, 1, self.bpsk)
        self.digital_diff_decoder_bb_0=digital.diff_decoder_bb(self.order)
        self.digital_crc32_async_bb_0=digital.crc32_async_bb(True)
        self.digital_costas_loop_cc_0=digital.costas_loop_cc(self.phase_bw, self.order, False)
        self.digital_correlate_access_code_xx_ts_0=digital.correlate_access_code_bb_ts("11100001010110101110100010010011",thresh, 'packet_len')
        self.digital_constellation_decoder_cb_0=digital.constellation_decoder_cb(self.bpsk)
        self.blocks_udp_sink_1=blocks.udp_sink(gr.sizeof_char*1, '127.0.0.1', self.UDP_port, 2048, True)
        self.blocks_throttle_0=blocks.throttle(gr.sizeof_gr_complex*1, self.samp_rate,True)
        self.blocks_tagged_stream_to_pdu_0=blocks.tagged_stream_to_pdu(blocks.byte_t, 'packet_len')
        self.blocks_repack_bits_bb_1=blocks.repack_bits_bb(1, 8, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_pdu_to_tagged_stream_0=blocks.pdu_to_tagged_stream(blocks.byte_t, 'packet_len')


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_tagged_stream_to_pdu_0, 'pdus'), (self.digital_crc32_async_bb_0, 'in'))
        self.msg_connect((self.digital_crc32_async_bb_0, 'out'), (self.blocks_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.blocks_pdu_to_tagged_stream_0, 0), (self.blocks_udp_sink_1, 0))
        self.connect((self.blocks_repack_bits_bb_1, 0), (self.blocks_tagged_stream_to_pdu_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_diff_decoder_bb_0, 0))
        self.connect((self.digital_correlate_access_code_xx_ts_0, 0), (self.blocks_repack_bits_bb_1, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_diff_decoder_bb_0, 0), (self.digital_map_bb_0, 0))
        self.connect((self.digital_lms_dd_equalizer_cc_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.digital_map_bb_0, 0), (self.digital_correlate_access_code_xx_ts_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_lms_dd_equalizer_cc_0, 0))
        self.connect((self.usrp_source, 0), (self.mmse_resampler_xx_0, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.blocks_throttle_0, 0))


    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate=usrp_rate
        self.mmse_resampler_xx_0.set_resamp_ratio(((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_thresh(self):
        return self.thresh

    def set_thresh(self, thresh):
        self.thresh=thresh

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps=sps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.mmse_resampler_xx_0.set_resamp_ratio(((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_rs_ratio(self):
        return self.rs_ratio

    def set_rs_ratio(self, rs_ratio):
        self.rs_ratio=rs_ratio
        self.mmse_resampler_xx_0.set_resamp_ratio(((self.usrp_rate/self.samp_rate)*self.rs_ratio))

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw=phase_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(self.phase_bw)
        self.digital_symbol_sync_xx_0.set_loop_bandwidth(self.phase_bw)

    def get_order(self):
        return self.order

    def set_order(self, order):
        self.order=order

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
    
def main(top_block_cls=packetReceive):

    tb = top_block_cls(
        samp_rate=600e3,
        sps=2,
        gain=60,
        freq=3.450e9,
        buffer_size=32768,
        bandwidth=20000000,
        SDR_ADDR="",
        UDP_port=40860
    )

    tb.start()
    time.sleep(20)
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
