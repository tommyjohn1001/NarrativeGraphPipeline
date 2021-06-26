from transformers import BertModel
import transformers
import torch.nn as torch_nn


transformers.logging.set_verbosity_error()


class BertBasedEmbedding(torch_nn.Module):
    """Embed and generate question-aware context"""

    def __init__(self, d_bert: int = 768, d_hid: int = 64, path_bert: str = None):
        super().__init__()

        self.d_bert = d_bert

        ## Modules for embedding
        self.bert_emb = BertModel.from_pretrained(path_bert)
        self.lin1 = torch_nn.Linear(d_bert, d_hid)

    def forward(self):
        return

    def encode_ques_para(self, ques_ids, context_ids, ques_mask, context_mask):
        # ques: [b, len_ques]
        # context_ids: [b, n_paras, len_para]
        # ques_mask: [b, len_ques]
        # context_mask: [b, n_paras, len_para]

        b, _, len_para = context_ids.shape

        #########################
        # Contextual embedding for question with BERT
        #########################
        ques = self.bert_emb(input_ids=ques_ids, attention_mask=ques_mask)[0]
        # [b, len_ques, d_bert]

        #########################
        # Contextual embedding for context with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        context = context_ids.view((-1, len_para))
        context_mask = context_mask.view((-1, len_para))
        # context     : [b*n_paras, len_para, d_bert]
        # context_mask: [b*n_paras, len_para]

        context = self.bert_emb(input_ids=context, attention_mask=context_mask)[0]
        # [b*n_paras, len_para, d_bert]
        context = context.view((b, -1, len_para, self.d_bert))
        # [b, n_paras, len_para, d_bert]

        ques, context = self.lin1(ques), self.lin1(context)

        return ques, context

    def encode_ans(self, input_ids):
        # input_ids : [b, len_]

        input_embds = self.bert_emb.embeddings.word_embeddings(input_ids)
        # [b, len_, d_bert]

        input_embds = self.lin1(input_embds)
        # [b, len_, d_hid]

        return input_embds

    def w_sum_ans(self, input_embds):
        # input_embds : [b, d_vocab]

        input_embds = input_embds @ self.bert_emb.embeddings.word_embeddings.weight
        # [b, d_bert]

        input_embds = self.lin1(input_embds)
        # [b, d_hid]

        return input_embds
