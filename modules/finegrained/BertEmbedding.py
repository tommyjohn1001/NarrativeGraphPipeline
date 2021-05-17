
from transformers import BertModel
import torch.nn as torch_nn
import torch

from configs import PATH, args


class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(PATH['bert'])
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, X, X_mask=None):
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

        self.embedding      = BertEmbedding()
        self.attn_mask_ques = torch_nn.Linear(args.seq_len_ques, args.seq_len_ques)
        self.attn_mask_paras= torch_nn.Linear(args.n_paras, args.n_paras)

        self.temperature    = 1.2

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
        _, ques_seq_embd = self.embedding(ques, ques_mask)
        # ques_seq_embd : [b, seq_len_ques, 768]

        ones        = torch.ones(b, args.seq_len_ques)
        attn_mask1  = torch.softmax(self.attn_mask_ques(ones), dim=1)
        # [b, seq_len_ques]

        # Apply attention mask for question
        attn_mask1  = attn_mask1.unsqueeze(2).repeat(1, 1, 768)
        ques_embd   = (ques_seq_embd * attn_mask1).sum(1)
        # [b, 768]


        ###################
        # Embed paragraphs
        ###################
        # Convert to another shape to fit with
        # input shape of self.embedding
        paras       = paras.reshape((-1, seq_len_para))
        paras_mask  = paras_mask.reshape((-1, seq_len_para))
        # paras, paras_mask: [b*n_paras, seq_len_para]

        para_embd, _    = self.embedding(paras, paras_mask)
        # [b*n_paras, 768]
        para_embd       = para_embd.reshape((b, -1, 768))
        # [b, n_paras, 768]


        # Dividing by temperature is a technique from topK temperature
        # By dividing with temperature > 1, we decrease the variance of softmax distribution
        attn_mask2  = torch.softmax(self.attn_mask_paras(ones)/self.temperature, dim=1)
        attn_mask2  = attn_mask2.unsqueeze(2).repeat(1, 1, 768)
        para_embd   = (para_embd * attn_mask2).sum(1)
        # [b, n_paras, 768]



        ###################
        # Embed answer
        ###################
        if ans:
            _, ans_seq_embd = self.embedding(ans, ans_mask)
            # ans_seq_embd : [b, seq_len_ans, 768]
        else:
            ans_seq_embd = None

        return ques_embd, para_embd, ans_seq_embd
