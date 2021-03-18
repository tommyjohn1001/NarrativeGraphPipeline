import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from modules.pg_decoder.utils import *
from configs import args


class PointerGeneratorDecoder(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.attn_pool  = AttentivePooling(args.dim_hid)
        # self.linear_dcd = torch_nn.Linear()

        # self.lstm       = torch_nn.LSTM()

    def forward(self, Y, H_q):
        # Y     : [batch, seq_len_quesm d_hid * 2]
        # H_q   : [batch, seq_len_ques, d_hid]

        pass
    