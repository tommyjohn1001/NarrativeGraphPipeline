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

    def encode_ques_para(self, ques_ids, context_ids, ques_mask, context_mask):
        # ques: [b, seq_len_ques]
        # context_ids: [b, n_paras, seq_len_para]
        # ques_mask: [b, seq_len_ques]
        # context_mask: [b, n_paras, seq_len_para]

        b, _, seq_len_para = context_ids.shape

        #########################
        # Contextual embedding for question with BERT
        #########################
        ques = self.bert_emb(input_ids=ques_ids, attention_mask=ques_mask)[0]
        # [b, seq_len_ques, d_bert]

        #########################
        # Contextual embedding for context with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        context = context_ids.view((-1, seq_len_para))
        context_mask = context_mask.view((-1, seq_len_para))
        # context     : [b*n_paras, seq_len_para, d_bert]
        # context_mask: [b*n_paras, seq_len_para]

        context = self.bert_emb(input_ids=context, attention_mask=context_mask)[0]
        # [b*n_paras, seq_len_para, d_bert]
        context = context.view((b, -1, seq_len_para, self.d_bert))
        # [b, n_paras, seq_len_para, d_bert]

        return ques, context

    def encode_ans(self, ans_ids, ans_mask):
        # ans_ids : [b, seq_len_ans]
        # ans_mask : [b, seq_len_ans]

        return self.bert_emb(input_ids=ans_ids, attention_mask=ans_mask)[0]
        # [b, seq_len_ans, d_bert]
