from transformers import BertModel
import transformers
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

transformers.logging.set_verbosity_error()


class FineGrain(torch_nn.Module):
    """Embed and generate question-aware context"""

    def __init__(
        self,
        len_para: int = 170,
        n_gru_layers: int = 5,
        d_bert: int = 768,
        path_bert: str = None,
    ):
        super().__init__()

        self.d_bert = d_bert

        self.bert_emb = BertModel.from_pretrained(path_bert)
        self.biGRU_CoAttn = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.biGRU_attn = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.biGRU_mask = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lin_attn = torch_nn.Linear(d_bert * 2, len_para)

    def forward(self):
        return

    def encode_ques_para(self, ques_ids, context_ids, ques_mask, context_mask):
        # ques_ids          : [b, len_ques]
        # context_ids         : [b, n_paras, len_para]
        # ques_mask     : [b, len_ques]
        # context_mask    : [b, n_paras, len_para]

        b, n_paras, len_para = context_ids.shape

        ques = self.bert_emb(input_ids=ques_ids, attention_mask=ques_mask)[0]
        # ques  : [b, len_ques, d_bert]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        paragraphs = []

        for ith in range(n_paras):
            contx = context_ids[:, ith, :]
            contx_mask = context_mask[:, ith, :]

            ###################
            # Embed context
            ###################
            L_s = self.bert_emb(input_ids=contx, attention_mask=contx_mask)[0]
            # L_s: [b, len_para, d_bert]

            ###################
            # Operate CoAttention between
            # query and context
            ###################

            # Affinity matrix
            A = torch.bmm(L_s, ques.transpose(1, 2))
            # A: [b, len_para, len_ques]

            # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
            S_q = torch.bmm(torch_f.softmax(A.transpose(1, 2), dim=1), L_s)
            # S_q: [b, len_ques, d_bert]

            X = torch.bmm(torch_f.softmax(A, dim=1), S_q)
            C_s = self.biGRU_CoAttn(X)[0]

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [b, 1, len_para, d_bert]

            paragraphs.append(C_s)

        context = torch.cat((paragraphs), dim=1)
        # [b, n_paras, len_para, d_bert]

        #########################
        # Reduce 'context' by applying attentive method
        # based on 'ques'
        #########################
        ques_ = torch.mean(ques, dim=1)
        # [b, d_bert]

        context_ = context.reshape(-1, len_para, self.d_bert)
        # paras_len = torch.sum(paras_mask, dim=2).reshape(-1).to("cpu")
        # context_    : [b*n_paras, len_para, d_bert]
        # paras_mask: [b*n_paras]

        # for i in range(paras_len.shape[0]):
        #     if paras_len[i] == 0:
        #         paras_len[i] = 1

        # tmp     = torch_nn.utils.rnn.pack_padded_sequence(context_, paras_len, batch_first=True,
        #                                                   enforce_sorted=False)
        # tmp     = self.biGRU_mask(tmp)[0]
        # context_  = torch_nn.utils.rnn.pad_packed_sequence(tmp, batch_first=True)[0]
        context_ = self.biGRU_mask(context_)[0]
        # [b*n_paras, len_para, d_bert]

        paras_first = context_[:, 0, :].reshape(b, n_paras, -1)
        # [b, n_paras, d_bert]

        q_ = ques_.unsqueeze(1).repeat(1, n_paras, 1)
        # [b, n_paras, d_bert]
        selector = torch.cat((q_, paras_first), dim=2)
        # [b, n_paras, d_bert*2]
        selector = self.lin_attn(selector)
        # [b, n_paras, len_para]

        selector = selector.unsqueeze(3).repeat(1, 1, 1, self.d_bert)
        # [b, n_paras, len_para, d_bert]

        context = torch.sum(context * selector, dim=2)
        # [b, n_paras, d_bert]

        return ques, context

    def get_w_embd(self, input_ids=None, input_embds=None):
        # input_ids : [b, len_]

        assert torch.is_tensor(input_ids) or torch.is_tensor(
            input_embds
        ), "One of two must not be None"

        if torch.is_tensor(input_ids):
            return self.bert_emb.embeddings.word_embeddings(input_ids)
        return input_embds @ self.bert_emb.embeddings.word_embeddings.weight
        # [b, len_, d_bert]

    def encode_ans(self, input_ids=None, input_embds=None, input_masks=None):
        # input_ids : [b, len_]
        # input_embds : [b, len_, d_bert]
        # input_mask : [b, len_]

        assert torch.is_tensor(input_ids) or torch.is_tensor(
            input_embds
        ), "One of two must not be None"

        encoded = (
            self.bert_emb(
                input_ids=input_ids,
                attention_mask=input_masks,
            )[0]
            if torch.is_tensor(input_ids)
            else self.bert_emb(
                inputs_embeds=input_embds,
                attention_mask=input_masks,
            )[0]
        )
        # [b, len_, d_bert]

        return encoded
