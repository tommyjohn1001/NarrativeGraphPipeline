import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from configs import args

Block   = 200

class IntrospectiveAlignmentLayer(torch_nn.Module):
    def __init__(self, d_hid=args.dim_hid, d_emb=200):
        super().__init__()

        self.d_hid  = d_hid
        self.d_emb  = d_emb


        self.biLSTM_emb     = torch_nn.LSTM(d_emb, d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)

        self.linearIAL      = torch_nn.Linear(d_hid, d_hid)

        self.linearReason   = torch_nn.Linear(4*d_hid, 4*d_hid)

        self.biLSTM_attn    = torch_nn.LSTM(8*d_hid, d_hid, num_layers=5,
                                           batch_first=True, bidirectional=True)

    def forward(self, ques, paras):
        # ques       : [batch, seq_len_ques, d_embd]
        # paras      : [batch, seq_len_contex, d_embd]

        batch, seq_len_context, _ = paras.shape

        ques    = torch_f.relu(ques)
        paras   = torch_f.relu(paras)

        # Input and Context embedding
        H_q = self.biLSTM_emb(ques)[0]
        H_c = self.biLSTM_emb(paras)[0]
        # H_q: [batch, seq_len_ques, d_hid]
        # H_c: [batch, seq_len_context, d_hid]


        # Introspective Alignment
        H_q = torch.sigmoid(self.linearIAL(H_q))
        H_c = torch.sigmoid(self.linearIAL(H_c))
        # H_q: [batch, seq_len_ques, d_hid]
        # H_c: [batch, seq_len_context, d_hid]

        E   = torch.bmm(H_c, torch.reshape(H_q, (batch, self.d_hid, -1)))
        # E: [batch, seq_len_context, seq_len_ques]
        A   = torch.bmm(torch_f.softmax(E, dim=1), H_q)
        # A: [batch, seq_len_context, d_hid]

        # Reasoning over alignments
        tmp = torch.cat((A, H_c, A - H_c, A * H_c), dim=-1)
        # tmp: [batch, seq_len_context, 4*d_hid]
        G   = self.linearReason(tmp)
        # G: [batch, seq_len_context, d_hid]

        result  = torch.zeros((batch, seq_len_context, seq_len_context))

        for b in range(batch):
            for i in range(seq_len_context):
                for j in range(seq_len_context):
                    if abs(i - j) <= Block:
                        result[b, i, j] = torch.matmul(G[b, i, :], G[b, j, :])
        G = result

        # Local BLock-based Self-Attention
        B = torch.bmm(torch_f.softmax(G, dim=1), tmp)
        # B: [batch, seq_len_context, 4*d_hid]

        Y = torch.cat((B, tmp), dim=-1)
        # Y: [batch, seq_len_context, 8*d_hid]
        Y = self.biLSTM_attn(Y)[0]
        # Y: [batch, seq_len_context, 2*d_hid]

        return Y
