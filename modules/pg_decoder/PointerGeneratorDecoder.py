
import torch.nn as torch_nn
import torch

from modules.utils import NonLinear, EmbeddingLayer
from modules.pg_decoder.utils import *
from configs import args


class PointerGeneratorDecoder(torch_nn.Module):
    def __init__(self):
        super().__init__()

        self.d_hid          = args.d_hid
        self.d_hid_PGD      = self.d_hid * 2
        self.d_vocab        = args.d_vocab
        self.n_layers       = args.n_layers
        self.seq_len_cntx   = args.seq_len_para * args.n_paras
        self.max_len_ans    = args.max_len_ans

        self.embedding      = EmbeddingLayer(d_hid=self.d_hid_PGD)

        self.linear_q       = torch_nn.Linear(self.d_hid, self.d_hid_PGD)
        self.attn_pool_q    = AttentivePooling(self.d_hid_PGD)
        self.linear_h1      = torch_nn.Linear(self.n_layers, self.seq_len_cntx)
        self.linear_h2      = NonLinear(self.d_hid_PGD, self.d_hid_PGD)
        self.linear_y       = NonLinear(self.d_hid_PGD, self.d_hid_PGD)


        self.linear_a       = torch_nn.Linear(self.d_hid_PGD, 1)

        self.lstm           = torch_nn.LSTM(self.d_hid_PGD * 2, self.d_hid_PGD,
                                            num_layers=args.n_layers, batch_first=True)
        self.linear_v1      = torch_nn.Linear(self.n_layers, 1)
        self.linear_v2      = torch_nn.Linear(self.d_hid_PGD, self.d_vocab)

        self.linear_pc1     = torch_nn.Linear(self.d_hid_PGD, 1, bias=False)
        self.linear_pc2     = torch_nn.Linear(self.n_layers, 1, bias=False)
        self.linear_ph1     = torch_nn.Linear(self.d_hid_PGD, 1, bias=False)
        self.linear_ph2     = torch_nn.Linear(self.n_layers, 1, bias=False)
        self.linear_py      = torch_nn.Linear(self.d_hid_PGD, 1)

    def forward(self, Y, H_q, trg, trg_mask):
        # Y         : [batch, seq_len_cntx, d_hid * 2]
        # H_q       : [batch, seq_len_ques, d_hid]
        # trg       : [batch, max_len_ans, d_embd]
        # trg_mask  : [batch, max_len_ans]

        batch   = Y.shape[0]

        #################################
        # Prepare initial tensors
        #################################
        def F_h(h_t):
            # h_t: [batch, n_layers, d_hid_PGD]

            h_t = self.linear_h1(torch.reshape(h_t, (batch, self.d_hid_PGD, -1)))
            # h_t: [batch, d_hid_PGD, seq_len_cntx]

            h_t = self.linear_h2(torch.reshape(h_t, (batch, -1, self.d_hid_PGD)))
            # h_t: [batch, seq_len_cntx, d_hid_PGD]

            return h_t

        def F_q(H_q):
            # H_q: [batch, seq_len_ques, d_hid]
            H_q = self.attn_pool_q(self.linear_q(H_q))
            # H_q: [batch, d_hid_PGD]

            H_q = H_q.unsqueeze(1).expand(batch, self.seq_len_cntx, self.d_hid_PGD)
            # H_q: [batch, seq_len_cntx, d_hid_PGD]

            return H_q

        def F_a(Y):
            # Y: [batch, seq_len_cntx, d_hid * 2]

            Y_  = self.linear_y(Y)
            # Y: [batch, seq_len_cntx, d_hid_PGD = d_hid*2]

            return Y_

        def F_v(h_t):
            # h_t: [batch, n_layers, d_hid_PGD]

            v_t = self.linear_v1(torch.reshape(h_t, (batch, self.d_hid_PGD, -1))).squeeze(-1)
            # v_t: [batch, d_hid_PGD]

            v_t = self.linear_v2(v_t)
            # v_t: [batch, d_vocab]

            return v_t


        Y_      = F_a(Y)
        H_q     = F_q(H_q)

        trg_    = self.embedding(trg)
        # trg_: [batch, max_len_ans, d_hid_PGD]

        # CLS is used to initialize LSTM
        cls_tok = torch.zeros((batch, 1, self.d_hid_PGD))

        h_t     = torch.zeros((batch, self.n_layers, self.d_hid_PGD))
        c_t     = torch.zeros((batch, self.n_layers, self.d_hid_PGD))

        pred    = torch.zeros((batch, self.max_len_ans))
        # pred: [batch, max_len_ans]

        #################################
        # For each timestep, infer answer word
        #################################
        for t in range(self.max_len_ans):
            ###################
            # Calculate attention
            ###################
            g   = torch.tanh(Y_+ F_h(h_t) + H_q)
            # g: [batch, seq_len_cntx, d_hid_PGD]
            a_t = self.linear_a(g)
            # a_t: [batch, seq_len_cntx, 1]
            y_t = torch.bmm(torch.reshape(a_t, (batch, -1, self.seq_len_cntx)), g)
            # y_t: [batch, 1, d_hid_PGD]

            ###################
            # Generate
            ###################
            if t == 0:
                tmp = torch.cat((y_t, cls_tok), dim=2)
            else:
                tmp = torch.cat((y_t, trg_[:,t-1,:].unsqueeze(1)), dim=2)
            # tmp: [batch, 1, d_hid_PGD * 2]
            _, (h_t, c_t)   = self.lstm(tmp, (h_t.reshape(self.n_layers, batch, -1),
                                              c_t.reshape(self.n_layers, batch, -1)))
            h_t = h_t.reshape((batch, self.n_layers, -1))
            c_t = c_t.reshape((batch, self.n_layers, -1))
            # h_t: [batch, n_layers, d_hid_PGD]
            # c_t: [batch, n_layers, d_hid_PGD]

            v_t = F_v(h_t)
            # v_t: [batch, d_vocab]

            ###################
            # Learn scalar switch
            ###################
            y_t     = self.linear_py(y_t).squeeze(-1)

            h_t_    = self.linear_ph2(self.linear_ph1(h_t).squeeze(-1))
            c_t_    = self.linear_pc2(self.linear_pc1(c_t).squeeze(-1))
            # h_t_, c_t_, y_t: [batch]
            p_t     = torch.sigmoid(h_t_ + c_t_ + y_t)
            # p_t: [batch]

            ###################
            # Predict word
            ###################
            # Pad a_t and v_t to d_vocab + seq_len_cntx
            padding = torch.zeros((batch, self.d_vocab))
            a_t     = torch.cat((a_t.squeeze(-1), padding), 1).unsqueeze(-1)
            # a_t: [batch, d_vocab + seq_len_cntx, 1]

            padding = torch.zeros((batch, self.seq_len_cntx))
            v_t     = torch.cat((v_t, padding), 1).unsqueeze(-1)
            # v_t: [batch, d_vocab + seq_len_cntx, 1]

            # Multiply a_t, v_t with p_t and 1 - p_t
            a_t     = torch.bmm(a_t, p_t.unsqueeze(-1)).squeeze(-1)
            v_t     = torch.bmm(v_t, (torch.ones((batch)) - p_t).unsqueeze(-1)).squeeze(-1)
            # a_t, v_t: [batch, d_vocab + seq_len_cntx]

            w_t     = torch.argmax(a_t + v_t, 1)
            # w_t: [batch]

            pred[:, t]  = w_t

        # Multiply 'pred' with 'trg_mask' to ignore masked position in tensor 'pred'
        pred    = torch.mul(pred, trg_mask)

        return pred
