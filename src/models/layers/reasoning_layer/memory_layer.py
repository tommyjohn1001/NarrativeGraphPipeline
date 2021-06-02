from typing import Any

import torch.nn as torch_nn
import torch

from src.models.layers.reasoning_layer.utils import CustomTransEnc, CustomTransEncLayer, Memory

class MemoryBasedReasoning(torch_nn.Module):
    def __init__(self,
        seq_len_ques: int = 42,
        seq_len_para: int = 122,
        seq_len_ans: int = 42,
        n_paras: int = 30,
        n_layers_gru: int = 5,
        n_heads_trans: int = 4,
        n_layers_trans: int = 3,
        d_hid: int = 64,
        d_bert: int = 768,
        device: Any = None):
        super().__init__()

        self.seq_len_ques   = seq_len_ques
        self.seq_len_para   = seq_len_para
        self.n_paras        = n_paras
        self.d_hid          = d_hid


        self.trans_enc  = CustomTransEnc(CustomTransEncLayer(d_model=d_hid, nhead=n_heads_trans),
                                         n_layers_trans, seq_len_ques, seq_len_para, d_hid)
        self.memory     = Memory(seq_len_ques, seq_len_para, n_layers_gru, d_hid, n_paras, device)

        self.lin1       = torch_nn.Linear(d_bert, d_hid, bias=False)
        self.lin2       = torch_nn.Linear(d_hid, seq_len_ans, bias=False)
        self.lin3       = torch_nn.Linear(seq_len_ques, (n_paras + 1) * (seq_len_ques + seq_len_para), bias=False)
        self.lin4       = torch_nn.Linear(d_hid, d_bert, bias=False)


    def forward(self, ques, paras, paras_mask):
        # ques      : [b, seq_len_ques, d_bert]
        # paras     : [b, n_paras, seq_len_para, d_bert]
        # paras_mask: [b, n_paras, seq_len_para]

        b, n_paras, _, _ = paras.shape

        ######################################
        # Transform input tensors to appropriate dimension
        ######################################
        ques        = self.lin1(ques)
        paras       = self.lin1(paras)
        # ques      : [b, seq_len_ques, d_bert]
        # paras     : [b, n_paras, seq_len_para, d_bert]


        ######################################
        # Reasoning with memory and TransformerEncoder
        ######################################

        ## Unsqueeze and repeat tensor 'ques' to match shape of 'paras'
        ques_        = ques.unsqueeze(1).repeat(1, self.n_paras, 1, 1)
        ques_paras   = torch.cat((ques_, paras), dim=2)
        # [b, n_paras, seq_len_ques + seq_len_para, d_hid]

        Y = []

        for nth_para in range(n_paras):
            ques_para   = ques_paras[:, nth_para]
            # [b, seq_len_ques + seq_len_para, d_hid]

            memory_cell = self.memory.get_mem_cell(nth_para)
            # [seq_len_ques + seq_len_para, d_hid]
            if torch.sum(memory_cell) == 0:
                memory_cell = ques_para
            else:
                memory_cell = memory_cell.repeat(b, 1, 1)
            # memory_cell: [b, seq_len_ques + seq_len_para, d_hid]

            output      = self.trans_enc(memory_cell.transpose(0, 1), ques_para.transpose(0, 1))
            # [b, seq_len_ques + seq_len_para, d_hid]

            Y.append(output.unsqueeze(1))


            self.memory.update_mem(nth_para, ques_para, output)


        final   = self.memory.get_final_memory(ques, paras, paras_mask)
        # [seq_len_ques + seq_len_para, d_hid]
        final   = final.reshape(1, 1, self.seq_len_ques + self.seq_len_para, self.d_hid).repeat(b, 1, 1, 1)

        Y.append(final)
        Y   = torch.cat(Y, dim=1).type_as(ques)
        # [b, n_paras + 1, seq_len_ques + seq_len_para, d_hid]


        ######################################
        # Derive attentive matrix from question
        # for tensor 'Y'
        ######################################
        ques    = self.lin2(ques)
        # [b, seq_len_ques, seq_len_ans]
        attentive = self.lin3(ques.transpose(1, 2))
        # [b, seq_len_ans, (n_paras + 1) * (seq_len_ques + seq_len_para)]

        Y       = Y.reshape(b, -1, self.d_hid)
        # [b, (n_paras + 1) * (seq_len_ques + seq_len_para), d_hid]

        Y       = torch.bmm(torch.softmax(attentive, dim=2), Y)
        # [b, seq_len_ans, d_hid]

        Y       = self.lin4(Y)
        # [b, seq_len_ans, d_bert]


        return Y
