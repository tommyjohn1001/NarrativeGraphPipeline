import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from modules.utils import NonLinear, transpose
from configs import args

Block   = 200

class IntrospectiveAlignmentLayer(torch_nn.Module):
    def __init__(self, d_hid=args.d_hid, d_emb=200):
        super().__init__()

        self.d_hid          = d_hid
        self.batch          = args.batch
        self.seq_len_contx  = args.seq_len_para * args.n_paras



        self.linearIAL      = NonLinear(d_hid, d_hid)

        self.linearReason   = torch_nn.Linear(4*d_hid, 4*d_hid)

        self.biLSTM_attn    = torch_nn.LSTM(8*d_hid, d_hid, num_layers=5,
                                           batch_first=True, bidirectional=True)


        self.mask           = torch.zeros((1, self.seq_len_contx, self.seq_len_contx))
        for i in range(self.seq_len_contx):
            for j in range(self.seq_len_contx):
                if abs(i - j) <= Block:
                    self.mask[0,i,j] = 1
        self.mask           = self.mask.repeat((self.batch, 1, 1)).to(args.device)

    def forward(self, H_q, H_c):
        # H_q   : [batch, seq_len_ques, d_hid]
        # H_c   : [batch, seq_len_contex, d_hid]


        # Introspective Alignment
        H_q = self.linearIAL(H_q)
        H_c = self.linearIAL(H_c)
        # H_q: [batch, seq_len_ques, d_hid]
        # H_c: [batch, x, d_hid]

        # Pad dim 1 of H_c
        pad_zeros = torch.zeros((H_c.shape[0], self.seq_len_contx - H_c.shape[1], self.d_hid)).to(args.device)
        H_c = torch.cat((H_c, pad_zeros), dim=1)

        # H_c: [batch, seq_len_contx, d_hid]

        E   = torch.bmm(H_c, transpose(H_q))
        # E: [batch, seq_len_contx, seq_len_ques]
        A   = torch.bmm(torch_f.softmax(E, dim=1), H_q)
        # A: [batch, seq_len_contx, d_hid]

        # Reasoning over alignments
        tmp = torch.cat((A, H_c, A - H_c, A * H_c), dim=-1)
        # tmp: [batch, seq_len_contx, 4*d_hid]
        G   = self.linearReason(tmp)
        # G: [batch, seq_len_contx, d_hid]


        G   = torch.bmm(G, transpose(G))

        G   = G * self.mask[:G.shape[0]]


        # Local BLock-based Self-Attention
        B = torch.bmm(torch_f.softmax(G, dim=1), tmp)
        # B: [batch, seq_len_contx, 4*d_hid]

        Y = torch.cat((B, tmp), dim=-1)
        # Y: [batch, seq_len_contx, 8*d_hid]
        Y = self.biLSTM_attn(Y)[0]
        # Y: [batch, seq_len_contx, 2*d_hid]

        return Y
