"""
Embedded Python Block
"""

import numpy as np
from gnuradio import gr
import pmt
import array
class blk(gr.sync_block):
    """Packet Format"""

    def __init__(self):
        gr.sync_block.__init__(self,
            name = "Packet Format GR38",
            in_sig = None,
            out_sig = None)
        self.message_port_register_in(pmt.intern('PDU_in'))
        self.message_port_register_out(pmt.intern('PDU_out0'))
        self.set_msg_handler(pmt.intern('PDU_in'), self.handle_msg)

    def handle_msg(self, msg):
        inMsg = pmt.to_python (msg)
        pld = inMsg[1] ## type-> numpy.ndarray
        mLen = len(pld)
        if (mLen > 0):
            ## create a numpy array of type 'int' with preamble and sync word
            tmp_char_list = np.array([85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,225,90,232,147],dtype=int)
            ## append length 2x
            tmp_char_list=np.append(tmp_char_list,(mLen >> 8))
            tmp_char_list=np.append(tmp_char_list,(mLen & 255))
            tmp_char_list=np.append(tmp_char_list,(mLen >> 8))
            tmp_char_list=np.append(tmp_char_list,(mLen & 255))
            tmp_char_list_len=len(tmp_char_list)
            ## append original payload
            new_char_list=np.insert(tmp_char_list,tmp_char_list_len,pld)
            new_char_list_len=len(new_char_list)
            ## save final numpy array as byte array (requires 'import array')
            byte_array_new_char_list=array.array('B',new_char_list)
            new_bytes_out_len = len(byte_array_new_char_list)
            ## create PMT u8vector using byte array
            new_out_bytes_pmt=pmt.cons(pmt.PMT_NIL,pmt.init_u8vector(new_bytes_out_len,(byte_array_new_char_list)))
            self.message_port_pub(pmt.intern('PDU_out0'), new_out_bytes_pmt)
