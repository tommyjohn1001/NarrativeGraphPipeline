
from transformers import BertModel
import transformers
import torch.nn as torch_nn


transformers.logging.set_verbosity_error()

class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self,
        path_bert: str = None,
    ):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(path_bert)

    def forward(self, X, X_mask=None):
        # X, X_mask: [b, *, d_bert]

        tmp = self.embedding(inputs_embeds=X, attention_mask=X_mask)

        return tmp[0]

class BertBasedLayer(torch_nn.Module):
    ''' Embed and generate question-aware context
    '''
    def __init__(self,
        d_vocab: int = 32716,
        d_bert: int = 768,
        path_bert: str = None):
        super().__init__()

        self.d_bert = d_bert

        ## Modules for embedding
        self.embd       = torch_nn.Embedding(d_vocab, d_bert)
        self.bert_emb   = BertEmbedding(path_bert)


    def forward(self):
        return

    def encode_ques_para(self, ques, paras, ques_mask, paras_mask):
        # ques          : [b, seq_len_ques]
        # paras         : [b, n_paras, seq_len_para]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]

        b, _, seq_len_para = paras.shape


        #########################
        # Convert ids to vector with nn.Embedding layer
        #########################
        ques    = self.embd(ques)
        paras   = self.embd(paras)
        # ques  : [b, seq_len_ques, d_bert]
        # paras : [b, n_paras, seq_len_para, d_bert]

        #########################
        # Contextual embedding for question with BERT
        #########################
        ques    = self.bert_emb(ques, ques_mask)
        # [b, seq_len_ques, d_bert]

        #########################
        # Contextual embedding for paras with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        paras       = paras.view((-1, seq_len_para, 768))
        paras_mask  = paras_mask.view((-1, seq_len_para))
        # paras     : [b*n_paras, seq_len_para, 768]
        # paras_mask: [b*n_paras, seq_len_para]

        paras   = self.bert_emb(paras, paras_mask)
        # [b*n_paras, seq_len_para, 768]
        paras   = paras.view((b, -1, seq_len_para, 768))
        # [b, n_paras, seq_len_para, d_bert]


        return ques, paras


    def encode_ans(self, ans, ans_mask):
        # ans           : [b, seq_len_ans]
        # ans_mask      : [b, seq_len_ans]

        ans = self.embd(ans)
        # [b, seq_len_ans, d_bert]
        return self.bert_emb(ans, ans_mask)
        # [b, seq_len_ans, d_bert]
