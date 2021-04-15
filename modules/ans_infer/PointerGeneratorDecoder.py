
import torch.nn as torch_nn
import torch

from modules.utils import NonLinear, EmbeddingLayer, transpose
from modules.ans_infer.utils import *
from configs import args


class PointerGeneratorDecoder(torch_nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.SEP_indx       = vocab.stoi['[SEP]']

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


    def get_mask_sep(self, pred):
        # X : [b, max_len_ans, d_vocab + seq_len_cntx]

        batch   = pred.shape[0]

        indx = torch.argmax(pred, dim=2)
        # [b, max_len_ans]


        SEP_indices = []
        for b in range(batch):
            for i in range(indx.shape[1]):
                if indx[b, i].item() == self.SEP_indx:
                    break
            SEP_indices.append(i)

        mask = []
        for b in range(batch):
            mask.append(torch.Tensor([1]*(SEP_indices[b]+1) +
                                    [0]*(self.max_len_ans - SEP_indices[b] - 1)).unsqueeze(0))

        mask = torch.vstack(mask).to(args.device)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.d_vocab + self.seq_len_cntx)

        return mask

    def forward(self, Y, H_q, ans, ans_len, ans_mask):
        # Y         : [b, seq_len_contx, d_hid * 2]
        # H_q       : [b, seq_len_ques, d_hid]
        # ans       : [b, max_len_ans, d_embd]
        # ans_len   : [b]
        # ans_mask  : [b, max_len_ans]

        batch   = Y.shape[0]

        #################################
        # Prepare initial tensors
        #################################
        def F_h(h_t):
            # h_t: [batch, n_layers, d_hid_PGD]

            h_t = self.linear_h1(transpose(h_t))
            # h_t: [batch, d_hid_PGD, seq_len_cntx]

            h_t = self.linear_h2(transpose(h_t))
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

            v_t = self.linear_v1(transpose(h_t)).squeeze(-1)
            # v_t: [batch, d_hid_PGD]

            v_t = self.linear_v2(v_t)
            # v_t: [batch, d_vocab]

            return v_t


        Y_      = F_a(Y)
        H_q     = F_q(H_q)

        ans_    = self.embedding(ans, ans_len)
        # ans_: [batch, max_len_ans, d_hid_PGD]

        # CLS is used to initialize LSTM
        cls_tok = torch.zeros((batch, 1, self.d_hid_PGD)).to(args.device)

        h_t     = torch.zeros((batch, self.n_layers, self.d_hid_PGD)).to(args.device)
        c_t     = torch.zeros((batch, self.n_layers, self.d_hid_PGD)).to(args.device)

        pred    = torch.zeros((batch, self.max_len_ans, self.d_vocab + self.seq_len_cntx)).to(args.device)
        # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]

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
            y_t = torch.bmm(transpose(a_t), g)
            # y_t: [batch, 1, d_hid_PGD]

            ###################
            # Generate
            ###################
            if t == 0:
                tmp = torch.cat((y_t, cls_tok), dim=2)
            else:
                tmp = torch.cat((y_t, ans_[:,t-1,:].unsqueeze(1)), dim=2)
            # tmp: [batch, 1, d_hid_PGD * 2]
            _, (h_t, c_t)   = self.lstm(tmp.contiguous(), (h_t.transpose(0, 1).contiguous(),
                                              c_t.transpose(0, 1).contiguous()))
            h_t = h_t.transpose(0, 1)
            c_t = c_t.transpose(0, 1)
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
            # h_t_, c_t_, y_t: [batch, 1]
            p_t     = torch.sigmoid(h_t_ + c_t_ + y_t).unsqueeze(-1)
            # p_t: [batch, 1, 1]

            ###################
            # Predict word
            ###################
            # Pad a_t and v_t to d_vocab + seq_len_cntx
            padding = torch.zeros((batch, self.d_vocab)).to(args.device)
            a_t     = torch.cat((a_t.squeeze(-1), padding), 1).unsqueeze(-1)
            # a_t: [batch, d_vocab + seq_len_cntx, 1]

            padding = torch.zeros((batch, self.seq_len_cntx)).to(args.device)
            v_t     = torch.cat((v_t, padding), 1).unsqueeze(-1)
            # v_t: [batch, d_vocab + seq_len_cntx, 1]

            # Multiply a_t, v_t with p_t and 1 - p_t
            a_t     = torch.bmm(a_t, p_t).squeeze(-1)

            v_t     = torch.bmm(v_t, (torch.ones((batch, 1, 1)).to(args.device) - p_t)).squeeze(-1)
            # a_t, v_t: [batch, d_vocab + seq_len_cntx]

            w_t     = a_t + v_t
            # w_t: [batch, d_vocab + seq_len_cntx]

            pred[:, t, :]   = w_t


        ########################
        # Pad 2nd dim of 'pred' from
        # max_len_ans to seq_len_ans
        ########################
        pad     = torch.zeros((batch, args.seq_len_ans - self.max_len_ans,
                               self.d_vocab + self.seq_len_cntx))
        pred    = torch.cat((pred, pad), dim=1)

        ########################
        # Multiply 'pred' with 2 masks
        ########################
        ans_mask    = ans_mask.unsqueeze(-1).repeat(1, 1, self.d_vocab + self.seq_len_cntx)
        # Multiply 'pred' with 'ans_mask' to ignore masked position in tensor 'pred'

        pred    = pred * ans_mask
        # pred: [batch, max_len_ans, d_vocab + seq_len_cntx]

        # Multiply 'pred' with mask SEP
        sep_mask = self.get_mask_sep(pred)
        pred    = pred * sep_mask
        # pred: [batch, seq_len_ans, d_vocab + seq_len_cntx]


        return pred
