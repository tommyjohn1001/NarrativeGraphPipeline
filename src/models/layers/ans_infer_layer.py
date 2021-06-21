import random
from typing import List

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch

from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class Decoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_vocab: int = 30522,
        d_hid: int = 64,
        cls_tok_id: int = 101,
        embd_layer: torch.nn.Module = None,
    ):
        super().__init__()

        self.len_ans = len_ans
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

        self.lin1 = torch_nn.Linear(d_hid, d_vocab)

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        teacher_forcing_ratio: float,
    ):
        # Y         : [b, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        final_dists = []
        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.len_ans + 1):
            ## Embed answer
            ans = self.embd_layer.encode_ans(input_ids, ans_mask[:, :ith])
            # [b, ith, d_hid]

            ## Decode answer
            decoder_output = self.trans_decoder(
                inputs_embeds=ans,
                attention_mask=ans_mask[:, :ith],
                encoder_hidden_states=Y,
            )[0][:, -1, :]
            # [b, d_hid]

            final = self.lin1(decoder_output)
            # [b, d_vocab]

            final_dists.append(final.unsqueeze(-1))

            ## Apply Scheduling teacher
            if ith == self.len_ans:
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
        # ans_ids   : [b, len_ans]
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
        # Y  : [n_paras, len_para, d_hid * 4]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids).type_as(Y).long().unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
