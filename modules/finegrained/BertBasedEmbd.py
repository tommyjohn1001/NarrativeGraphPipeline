
from transformers import BertModel
import torch.nn as torch_nn

from configs import args


class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(args.bert_model)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, X, X_mask=None):
        # X, X_mask: [b, seq_len]

        tmp = self.embedding(inputs_embeds=X,
                             attention_mask=X_mask)

        return tmp[0]

class BertBasedEmbd(torch_nn.Module):
    """Thid module employs Bert model to embed question, paras and answers."""

    def __init__(self):
        super().__init__()

        self.embedding  = BertEmbedding()
        self.linear1    = torch_nn.Linear(200, 768)
        self.linear2    = torch_nn.Linear(768, args.d_hid)

    def forward(self, ques, paras, ans, ques_mask, paras_mask, ans_mask):
        # ques          : [b, seq_len_ques, 200]
        # paras         : [b, n_paras, seq_len_para, 200]
        # ans           : [b, seq_len_ans, 200]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]
        # ans_mask      : [b, seq_len_ans]

        b, _, seq_len_para, _ = paras.shape

        ###################
        # Convert ques, paraas and
        # ans to new space
        ###################
        ques    = self.linear1(ques)
        paras   = self.linear1(paras)
        ans     = self.linear1(ans)
        # ques  : [b, seq_len_ques, 768]
        # paras : [b, seq_len_contx, 768]
        # ans   : [b, seq_len_ans, 768]

        ###################
        # Embed question
        ###################
        ques    = self.embedding(ques, ques_mask)
        # [b, seq_len_ques, 768]


        ###################
        # Embed paragraphs
        ###################
        # Convert to another shape to fit with
        # input shape of self.embedding
        paras       = paras.view((-1, seq_len_para, 768))
        paras_mask  = paras_mask.view((-1, seq_len_para))
        # paras     : [b*n_paras, seq_len_para, 768]
        # paras_mask: [b*n_paras, seq_len_para]

        print(paras.shape)

        paras   = self.embedding(paras, paras_mask)
        # [b*n_paras, seq_len_para, 768]
        paras   = paras.view((b, -1, 768))
        # [b, seq_len_contx=n_paras*seq_len_para, 768]


        ###################
        # Embed answer
        ###################
        ans     = self.embedding(ans, ans_mask)
        # ans   : [b, seq_len_ans, 768]


        ###################
        # Convert ques, paraas and
        # ans to new space
        ###################
        ques    = self.linear2(ques)
        paras   = self.linear2(paras)
        ans     = self.linear2(ans)
        # ques  : [b, seq_len_ques, d_hid]
        # paras : [b, seq_len_contx, d_hid]
        # ans   : [b, seq_len_ans, d_hid]

        return ques, paras, ans
