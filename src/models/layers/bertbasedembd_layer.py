from transformers import BertModel
import transformers
import torch.nn as torch_nn


transformers.logging.set_verbosity_error()


class BertBasedEmbedding(torch_nn.Module):
    """Embed and generate question-aware context"""

    def __init__(self, d_bert: int = 768, path_bert: str = None):
        super().__init__()

        self.d_bert = d_bert

        ## Modules for embedding
        self.bert_emb = BertModel.from_pretrained(path_bert)

    def forward(self):
        return

    def encode_ques_para(self, ques, paras, ques_mask, paras_mask):
        # ques          : [b, seq_len_ques]
        # paras         : [b, n_paras, seq_len_para]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]

        b, _, seq_len_para = paras.shape

        #########################
        # Contextual embedding for question with BERT
        #########################
        ques = self.bert_emb(input_ids=ques, attention_mask=ques_mask)[0]
        # [b, seq_len_ques, d_bert]

        #########################
        # Contextual embedding for paras with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        paras = paras.view((-1, seq_len_para))
        paras_mask = paras_mask.view((-1, seq_len_para))
        # paras     : [b*n_paras, seq_len_para, d_bert]
        # paras_mask: [b*n_paras, seq_len_para]

        paras = self.bert_emb(input_ids=paras, attention_mask=paras_mask)[0]
        # [b*n_paras, seq_len_para, d_bert]
        paras = paras.view((b, -1, seq_len_para, self.d_bert))
        # [b, n_paras, seq_len_para, d_bert]

        return ques, paras

    def encode_ans(self, ans, ans_mask):
        # ans           : [b, seq_len_ans]
        # ans_mask      : [b, seq_len_ans]

        return self.bert_emb(input_ids=ans, attention_mask=ans_mask)[0]
        # [b, seq_len_ans, d_bert]
