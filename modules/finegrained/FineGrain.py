
from transformers import BertModel
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from modules.utils import transpose
from configs import args

class FineGrain(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self):
        super().__init__()

        self.d_hid      = args.d_hid
        self.d_emb_bert = 768

        ## Modules for embedding
        self.embedding      = BertModel.from_pretrained(args.bert_model)
        self.biGRU_emb      = torch_nn.GRU(self.d_emb_bert, self.d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)
        self.linear_embd    = torch_nn.Sequential(
            torch_nn.Linear(self.d_emb_bert, self.d_emb_bert),
            torch_nn.ReLU(),
            torch_nn.Dropout(args.dropout)
        )

        self.biGRU_CoAttn   = torch_nn.GRU(self.d_hid, self.d_hid//2, num_layers=5,
                                       batch_first=True, bidirectional=True)

    def forward(self, ques, paras, ques_mask, paras_mask):
        """ As implementing this module (Mar 11), it is aimed runing for each pair
        of question and each paragraph
        """
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        batch, seq_len_ques = ques.shape
        n_paras             = paras.shape[1]

        question    = torch.zeros((batch, seq_len_ques, self.d_hid)).to(args.device)
        paragraphs  = None

        for ith in range(n_paras):
            para        = paras[:, ith, :]
            para_mask   = paras_mask[:, ith, :]

            ###################
            # Embed query and context
            ###################
            L_q = self.embedding(ques, ques_mask)[0]
            L_s = self.embedding(para, para_mask)[0]
            # L_q: [b, seq_len_ques, d_embd]
            # L_s: [b, seq_len_para, d_embd]

            L_q = torch.tanh(self.linear_embd(L_q))

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
            # C_s: [b, 1, seq_len_para, d_hid]


            question += S_q
            if paragraphs is None:
                paragraphs = C_s
            else:
                paragraphs = torch.cat((paragraphs, C_s), dim=1)

        # question  : [b, seq_len_ques, d_hid]
        # paragraphs: [b, n_paras, seq_len_para, d_hid]

        paragraphs  = paragraphs.view(batch, -1, self.d_hid)
        # [b, seq_len_contx=n_paras*seq_len_para, d_hid]

        return question, paragraphs
