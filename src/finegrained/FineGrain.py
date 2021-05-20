
from transformers import BertModel
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from src.utils import transpose
from configs import args, PATH


class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(PATH['bert'])
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, X, X_mask=None):
        # X, X_mask: [b, seq_len]

        tmp = self.embedding(inputs_embeds=X, attention_mask=X_mask)

        return tmp[0]

class FineGrain(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self):
        super().__init__()

        d_hid  = args.d_hid
        d_emb  = args.d_embd

        ## Modules for embedding
        self.embedding      = BertEmbedding()
        self.biGRU_emb      = torch_nn.GRU(d_emb, d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)
        self.linear_embd    = torch_nn.Sequential(
            torch_nn.Linear(d_emb, d_emb),
            torch_nn.ReLU()
        )

        self.biGRU_CoAttn   = torch_nn.GRU(d_hid, d_hid//2, num_layers=5,
                                       batch_first=True, bidirectional=True)

        self.linear_ans     = torch_nn.Linear(768, d_hid, bias=False)

    def forward(self, ques, paras, ans, ques_mask, paras_mask, ans_mask):
        # ques          : [b, seq_len_ques, 200]
        # paras         : [b, n_paras, seq_len_para, 200]
        # ans           : [b, seq_len_ans, 200]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]
        # ans_mask      : [b, seq_len_ans]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        b, seq_len_ques, _ = ques.shape
        n_paras            = paras.shape[1]

        question    = torch.zeros((b, seq_len_ques, self.d_hid)).to(args.device)
        paragraphs  = None

        ques = self.embedding(ques, ques_mask)[0]
        for ith in range(n_paras):
            para        = paras[:, ith, :]
            para_mask   = paras_mask[:, ith, :]

            ###################
            # Embed query and context
            ###################
            L_q = ques
            L_s = self.embedding(para, para_mask)[0]
            # L_q: [b, seq_len_ques, 768]
            # L_s: [b, seq_len_para, 768]

            L_q = self.linear_embd(L_q)

            E_q = self.biGRU_emb(L_q)[0]
            E_s = self.biGRU_emb(L_s)[0]

            # E_q: [b, seq_len_ques, d_hid]
            # E_s: [b, seq_len_para, d_hid]


            ###################
            # Operate CoAttention between
            # query and context
            ###################

            # Affinity matrix
            A   = torch.bmm(E_s, transpose(E_q))
            # A: [b, seq_len_para, seq_len_ques]

            # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
            S_q = torch.bmm(torch_f.softmax(transpose(A), dim=1), E_s)
            # S_q: [b, seq_len_ques, d_hid]


            X   = torch.bmm(torch_f.softmax(A, dim=1), S_q)
            C_s = self.biGRU_CoAttn(X)[0]
            C_s = transpose(C_s)

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [b, 1, seq_len_ques, d_hid]


            question += S_q
            if paragraphs is None:
                paragraphs = C_s
            else:
                paragraphs = torch.cat((paragraphs, C_s), dim=1)

        # question  : [b, seq_len_ques, d_hid]
        # paragraphs: [b, n_paras, seq_len_ques, d_hid]


        ###################
        # Embed answer
        ###################
        if ans:
            _, answer   = self.embedding(ans, ans_mask)
            answer      = self.linear_ans(answer)
            # [b, seq_len_ans, d_hid]
        else:
            answer = None

        return question, paragraphs, answer
