
from transformers import BertModel
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from configs import args

class FineGrain(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self, d_hid=256, d_emb=768):
        super().__init__()

        ## Modules for embedding
        self.embedding      = BertModel.from_pretrained(args.bert_model)
        self.biGRU_emb      = torch_nn.GRU(d_emb, d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)
        self.linear_embd    = torch_nn.Linear(d_emb, d_emb)

        self.biGRU_CoAttn   = torch_nn.GRU(d_hid, d_hid//2, num_layers=5,
                                       batch_first=True, bidirectional=True)


    def forward(self, ques, para, ques_mask, para_mask):
        """ As implementing this module (Mar 11), it is aimed runing for each pair
        of question and each paragraph
        """
        # ques : [batch, seq_len_ques]
        # ques_mask : [batch, seq_len_ques]
        # para : [batch, seq_len_para]
        # para_mask : [batch, seq_len_para]
        batch, seq_len_ques = ques.shape
        seq_len_para        = para.shape[1]

        ###################
        # Embed query and context
        ###################
        L_q = self.embedding(ques, ques_mask)[0]
        L_s = self.embedding(para, para_mask)[0]
        # L_q: [batch, seq_len_ques, d_embd]
        # L_s: [batch, seq_len_para, d_embd]

        L_q = torch.tanh(self.linear_embd(L_q))

        E_q = self.biGRU_emb(L_q)[0]
        E_s = self.biGRU_emb(L_s)[0]

        # E_q: [batch, seq_len_ques, d_hid]
        # E_s: [batch, seq_len_para, d_hid]


        ###################
        # Operate CoAttention between
        # query and context
        ###################

        # Affinity matrix
        A   = torch.bmm(E_s, torch.reshape(E_q, (batch, -1, seq_len_ques)))
        # A: [batch, seq_len_para, seq_len_ques]

        # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
        S_q = torch.bmm(torch_f.softmax(torch.reshape(A, (batch, seq_len_ques, seq_len_para)), dim=1), E_s)
        # S_q: [batch, seq_len_ques, d_hid]


        X   = torch.bmm(torch_f.softmax(A, dim=1), S_q)
        C_s = self.biGRU_CoAttn(X)[0]
        C_s = torch.reshape(C_s, (batch, seq_len_para, -1))

        return C_s, S_q
