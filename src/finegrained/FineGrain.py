
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
        # X, X_mask: [b, *, 768]

        tmp = self.embedding(inputs_embeds=X, attention_mask=X_mask)

        return tmp[0]

class FineGrain(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self):
        super().__init__()

        self.d_hid  = args.d_hid
        d_emb       = args.d_embd

        ## Modules for embedding
        self.embedding      = BertEmbedding()
        self.linear1        = torch_nn.Linear(d_emb, 768, bias=False)
        self.linear_ans     = torch_nn.Linear(768, self.d_hid, bias=False)

        self.biGRU_emb      = torch_nn.GRU(768, self.d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)
        self.linear_embd    = torch_nn.Sequential(
            torch_nn.Linear(768, 768),
            torch_nn.ReLU(),
            torch_nn.Dropout(args.dropout)
        )

        self.biGRU_CoAttn   = torch_nn.GRU(self.d_hid, self.d_hid//2, num_layers=5,
                                       batch_first=True, bidirectional=True)


    def forward(self, ques, paras, ans, ques_mask, paras_mask, ans_mask):
        # ques          : [b, seq_len_ques, 200]
        # paras         : [b, n_paras, seq_len_para, 200]
        # ans           : [b, seq_len_ans, 200]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]
        # ans_mask      : [b, seq_len_ans]

        b, seq_len_ques, _ = ques.shape
        n_paras            = paras.shape[1]


        ques    = self.linear1(ques.float())
        paras   = self.linear1(paras.float())
        ans     = self.linear1(ans.float())
        # ques          : [b, seq_len_ques, 768]
        # paras         : [b, n_paras, seq_len_para, 768]
        # ans           : [b, seq_len_ans, 768]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        question    = torch.zeros((b, seq_len_ques, self.d_hid)).to(args.device)
        paragraphs  = []

        ques = self.embedding(ques, ques_mask)
        for ith in range(n_paras):
            para        = paras[:, ith, :, :]
            para_mask   = paras_mask[:, ith, :]

            ###################
            # Embed query and context
            ###################
            L_q = ques
            L_s = self.embedding(para, para_mask)
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

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [b, 1, seq_len_para, d_hid]


            question += S_q
            paragraphs.append(C_s)

        paragraphs = torch.cat((paragraphs), dim=1)

        # question  : [b, seq_len_ques, d_hid]
        # paragraphs: [b, n_paras, seq_len_para, d_hid]


        ###################
        # Embed answer
        ###################
        if torch.is_tensor(ans):
            answer   = self.encode_ans(ans, ans_mask)
            # [b, seq_len_ans, d_hid]
        else:
            answer = None

        return question, paragraphs, answer

    def encode_ans(self, ans, ans_mask):
        # ans           : [b, seq_len_ans, 200]
        # ans_mask      : [b, seq_len_ans]

        return self.linear_ans(self.embedding(ans, ans_mask))
        # [b, seq_len_ans, d_hid]
