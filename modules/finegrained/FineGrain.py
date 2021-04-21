
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from modules.finegrained.BertEmbedding import BertEmbedding
from modules.utils import transpose
from configs import args

class FineGrain(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self, d_hid=256, d_hid2=512, d_emb=768):
        super().__init__()

        self.d_hid  = d_hid
        self.d_hid2 = d_hid2
        self.d_emb  = d_emb

        ## Modules for embedding
        self.embedding      = BertEmbedding()
        self.biGRU_emb      = torch_nn.GRU(d_emb, d_hid//2, num_layers=5,
                                           batch_first=True, bidirectional=True)
        self.linear_embd    = torch_nn.Linear(d_emb, d_emb)

        self.biGRU_CoAttn   = torch_nn.GRU(d_hid, d_hid//2, num_layers=5,
                                       batch_first=True, bidirectional=True)

        self.linear_convert = torch_nn.Linear(d_hid, d_hid2)

    def forward(self, ques, paras, ques_mask, paras_mask):
        """ As implementing this module (Mar 11), it is aimed runing for each pair
        of question and each paragraph
        """
        # ques       : [batch, seq_len_ques]
        # ques_mask  : [batch, seq_len_ques]
        # paras      : [batch, n_paras, seq_len_para]
        # paras_mask : [batch, n_paras, seq_len_para]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        seq_len_ques    = args.seq_len_ques
        batch           = args.batch
        n_paras         = args.n_paras

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
            # L_q: [batch, seq_len_ques, 768]
            # L_s: [batch, seq_len_para, 768]

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
            A   = torch.bmm(E_s, transpose(E_q))
            # A: [batch, seq_len_para, seq_len_ques]

            # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
            S_q = torch.bmm(torch_f.softmax(transpose(A), dim=1), E_s)
            # S_q: [batch, seq_len_ques, d_hid]


            X   = torch.bmm(torch_f.softmax(A, dim=1), S_q)
            C_s = self.biGRU_CoAttn(X)[0]
            C_s = transpose(C_s)

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [batch, 1, seq_len_ques, d_hid]


            question += S_q
            if paragraphs is None:
                paragraphs = C_s
            else:
                paragraphs = torch.cat((paragraphs, C_s), dim=1)

        # question  : [batch, seq_len_ques, d_hid]
        # paragraphs: [batch, n_paras, seq_len_ques, d_hid]

        #########################
        # Convert to final 'paragraphs' tensor
        #########################
        question    = torch.sum(question, dim=1)
        question    = torch.unsqueeze(question, 1)
        # question  : [batch, 1, d_hid]

        paragraphs  = torch.sum(paragraphs, dim=2)
        # paragraphs: [batch, n_paras, d_hid]

        paragraphs  = torch.cat((paragraphs, question), dim=1)
        # paragraphs: [batch, n_paras+1, d_hid]
        paragraphs  = self.linear_convert(paragraphs)
        # paragraphs: [batch, n_paras+1, d_hid2]

        return paragraphs
