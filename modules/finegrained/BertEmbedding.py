
from transformers import BertModel
import torch.nn as torch_nn

from configs import args


class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(args.bert_model)

    def forward(self, X, X_mask):
        # X, X_mask: [b, seq_len]

        tmp = self.embedding(X, X_mask)
        X, X_len = tmp[1], tmp[0]  
        # X: [b, 768]
        # X: [b, seq_len, 768]

        return X, X_len

class SimpleBertEmbd(torch_nn.Module):
    """Thid module employs Bert model to embed question, paras and answers."""

    def __init__(self):
        super().__init__()

        self.embedding  = BertEmbedding()

    def forward(self, ques, ques_mask, paras, paras_mask, ans, ans_mask):
        # ques          : [b, seq_len_ques]
        # ques_mask     : [b, seq_len_ques]
        # paras         : [b, n_paras, seq_len_para]
        # paras_mask    : [b, n_paras, seq_len_para]
        # ans           : [b, seq_len_ans]
        # ans_mask      : [b, seq_len_ans]

        b, _, seq_len_para = paras.shape

        ###################
        # Embed question
        ###################
        ques_embd, ques_seq_embd = self.embedding(ques, ques_mask)
        # ques_embd     : [b, 768]
        # ques_seq_embd : [b, seq_len_ques, 768]


        ###################
        # Embed paragraphs
        ###################
        # Convert to another shape to fit with
        # input shape of self.embedding
        paras       = paras.reshape((-1, seq_len_para))
        paras_mask  = paras_mask.reshape((-1, seq_len_para))
        # paras, paras_mask: [b*n_paras, seq_len_para]

        _, paras_seq_embd   = self.embedding(paras, paras_mask)
        # [b*n_paras, 768]
        paras_seq_embd      = paras_seq_embd.reshape((b, -1, 768))
        # [b, n_paras, 768]


        ###################
        # Embed answer
        ###################
        _, ans_seq_embd = self.embedding(ans, ans_mask)
        # ans_seq_embd : [b, seq_len_ans, 768]

        return ques_embd, ques_seq_embd, paras_seq_embd, ans_seq_embd
