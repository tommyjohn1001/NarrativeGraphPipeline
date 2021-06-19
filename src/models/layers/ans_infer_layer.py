import random
from typing import List

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch

from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        seq_len_para: int = 170,
        seq_len_ans: int = 15,
        d_vocab: int = 30522,
        n_paras: int = 5,
        d_bert: int = 768,
        d_hid: int = 64,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.seq_len_ans = seq_len_ans
        self.d_hid = d_hid
        self.cls_tok_id = cls_tok_id
        self.d_vocab = d_vocab

        self.d_hid_ = d_hid * 4
        self.embd_layer: BertBasedEmbedding = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.hidden_size = d_hid
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Linear(d_bert, d_hid)
        self.lin_attn_1 = torch_nn.Linear(self.d_hid_, self.d_hid_)
        self.lin_attn_2 = torch_nn.Linear(d_hid, self.d_hid_)
        self.lin_attn_3 = torch_nn.Linear(self.d_hid_, 1)
        self.ff_attn_1 = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid_, self.d_hid_),
            torch_nn.BatchNorm1d(n_paras * seq_len_para),
            torch_nn.Tanh(),
        )
        self.lin_abstr_1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid * 5, d_hid * 5),
            torch_nn.Linear(d_hid * 5, d_vocab),
        )

        self.lin_swch_1 = torch_nn.Linear(d_hid * 5, 1)
        self.lin_swch_2 = torch_nn.Linear(d_hid, 1)
        self.lin_swch_3 = torch_nn.Linear(d_hid, 1)

    def forward(self, ans: torch.Tensor, ans_mask: torch.Tensor):
        # ans      : [b, seq_len_ans, d_hid]
        # ans_mask : [b, seq_len_ans]

        output = self.trans_decoder(inputs_embeds=ans, attention_mask=ans_mask)[0]
        # [b, seq_len_ans, d_hid]

        return output

    def attention(self, Y: torch.Tensor, decoder_out: torch.Tensor):
        # Y: [b, n_paras, seq_len_para, d_hid_]
        # decoder_out: [b, d_hid]

        Y_ = self.lin_attn_1(Y)
        # [b, seq_len_contx, d_hid_]
        decoder_out = self.lin_attn_2(decoder_out).unsqueeze(1)
        # [b, 1, d_hid_]

        e_t = self.ff_attn_1(Y_ + decoder_out)
        # [b, seq_len_contx, d_hid_]
        e_t = self.lin_attn_3(e_t)
        # [b, seq_len_contx, 1])

        extract_dist = torch.softmax(e_t, dim=1)

        return extract_dist

    def pointer_generator(self, Y, context_ids, decoder_output, ans):
        # Y             : [b, seq_len_contx, d_hid * 4]
        # context_ids   : [b, seq_len_contx]
        # decoder_output: [b, d_hid]
        # ans           : [b, ith, d_hid]

        #########################
        ## Calculate extractive dist over context
        #########################
        extract_dist = self.attention(Y, decoder_output)
        # [b, seq_len_contx, 1]

        #########################
        ## Calculate abstractive dist over gen vocab
        #########################
        contx = (extract_dist.repeat(1, 1, Y.size()[-1]) * Y).sum(dim=1)
        # [b, d_hid_]
        contx = torch.cat((contx, decoder_output), dim=-1)
        # [b, d_hid * 5]
        abstr_dist = self.lin_abstr_1(contx)
        # [b, d_vocab]

        #########################
        ## Calculate extract-abstract switch and combine them
        #########################
        switch = (
            self.lin_swch_1(contx)
            + self.lin_swch_2(ans[:, -1])
            + self.lin_swch_3(decoder_output)
        )
        # [b, 1]
        switch = torch.sigmoid(switch)

        final = switch * abstr_dist
        # [b, d_vocab]

        # Scatter
        extract_dist = (1 - switch) * extract_dist.squeeze(-1)
        final = final.scatter_add(
            dim=-1, index=context_ids, src=extract_dist.type_as(final)
        )
        # [b, d_vocab]

        return final

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        context_ids: torch.Tensor,
        teacher_forcing_ratio: float,
    ):
        # Y         : [b, n_paras, seq_len_para, d_hid * 4]
        # ans_ids   : [b, seq_len_ans]
        # ans_mask  : [b, seq_len_ans]
        # context_ids: [b, n_paras, seq_len_para]

        b = Y.size()[0]
        Y = Y.view(b, -1, self.d_hid * 4)
        context_ids = context_ids.view(b, -1)

        final_dists = []
        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.seq_len_ans + 1):
            ## Embed answer
            ans = self.embd_layer.encode_ans(input_ids, ans_mask[:, :ith])
            # [b, ith, d_bert]
            ans = self.lin1(ans)
            # [b, ith, d_hid]

            ## Decode answer
            decoder_output = self(ans, ans_mask[:, :ith])[:, -1, :]
            # [b, d_hid]

            ## Get final distribution over extended vocab
            final = self.pointer_generator(
                Y=Y, context_ids=context_ids, decoder_output=decoder_output, ans=ans
            )
            # [b, d_vocab]

            final_dists.append(final.unsqueeze(-1))

            ## Apply Scheduling teacher
            if ith == self.seq_len_ans:
                break

            final = torch.argmax(final, dim=-1)
            chosen = self.choose_teacher_forcing(
                teacher_forcing_ratio=teacher_forcing_ratio,
                output=final,
                ans_ids=ans_ids,
                ith=ith,
            )
            # [b]
            input_ids = torch.cat((input_ids, chosen.detach().unsqueeze(1)), dim=1)

        return torch.cat(final_dists, dim=-1)

    def choose_teacher_forcing(
        self,
        teacher_forcing_ratio: float,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        ith: int,
    ):
        # output: [b]
        # ans_ids   : [b, seq_len_ans]
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        return output if use_teacher_forcing else ans_ids[:, ith]

    def do_predict(
        self,
        decoder_input_ids: List,
        Y: torch.Tensor,
        context_ids: torch.Tensor,
    ):
        ## TODO: Finish implementation later

        # decoder_input_ids: list of indices
        # Y  : [n_paras, seq_len_para, d_hid * 4]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids).type_as(Y).long().unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
