from typing import Any, List


from transformers import BertConfig, BertModel
from torch.nn.parameter import Parameter
import torch.nn as torch_nn
import torch
import numpy as np


from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class Decoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_vocab: int = 30522,
        d_hid: int = 64,
        cls_tok_id: int = 101,
        pad_tok_id: int = 0,
        embd_layer: torch.nn.Module = None,
        max_step: int = 0,
    ):
        super().__init__()

        self.len_ans = len_ans
        self.d_hid = d_hid
        self.cls_tok_id = cls_tok_id
        self.pad_tok_id = pad_tok_id
        self.d_vocab = d_vocab
        self.epsilon = 1
        self.bigEPSILON = 0.5
        self.max_step = max_step

        self.embd_layer: BertBasedEmbedding = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.hidden_size = d_hid
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Linear(d_hid, d_vocab)

    def forward(
        self,
        Y: torch.Tensor,
        input_embds: torch.Tensor,
        input_masks: torch.Tensor,
    ):
        # Y          : [b, len_para, d_hid]
        # input_embds: [b, len_, d_vocab]
        # input_masks: [b, len_]

        input_embds = self.embd_layer.encode_ans(input_embds)
        # [b, d_hid]

        output = self.trans_decoder(
            inputs_embeds=input_embds,
            attention_mask=input_masks,
            encoder_hidden_states=Y,
        )[0]
        # [b, len_, d_hid]

        output = self.lin1(output)
        # [b, len_, d_vocab]

        return output

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        cur_step: int,
        cur_epoch: int,
    ):
        # Y         : [b, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        b = Y.size(0)

        input_emds = []

        ## Init input_embs with cls_embd
        cls_embd = self.get_embd_tensor(b=b, device=Y.device, ids=self.cls_tok_id)
        # [b, d_vocab]

        input_emds.append(cls_embd.unsqueeze(1))

        for ith in range(1, self.len_ans + 1):
            output = self(
                Y=Y,
                input_embds=torch.cat(input_emds, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            ## Apply Scheduling teacher
            output_tok_emb = self.choose_scheduled_sampling(
                output=output,
                ans_ids=ans_ids,
                cur_step=cur_step,
                cur_epoch=cur_epoch,
            )
            # [b, d_vocab]
            input_emds.append(output_tok_emb.unsqueeze(1))

        return torch.cat(input_emds, dim=1)[:, :-1].transpose(1, 2)

    ##############################################
    # Methods for scheduled sampling
    ##############################################
    def get_embd_tensor(self, b: int, device: Any, ids: Any):
        embd = torch.zeros((b, self.d_vocab), device=device)
        # [b, d_vocab]

        if not torch.is_tensor(ids):
            assert isinstance(ids, int)
            ids = torch.full((b, 1), fill_value=ids, device=device)
        ones = torch.ones((b, 1), device=device)
        embd = torch.scatter_add(embd, dim=1, index=ids, src=ones)
        # [b, d_vocab]

        return embd

    def choose_scheduled_sampling(
        self,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        cur_step: int,
        cur_epoch: int,
    ):
        # output: [b, len_, d_vocab]
        # ans_ids: [b, len_ans]

        b = output.size(0)

        ith = output.size(1)
        output = output[:, -1, :]
        # [b, d_vocab]

        ## Create tensor full of PAD indices if ith == len_ans - 1
        pad_embd = None
        if ith == self.len_ans:
            pad_embd = self.get_embd_tensor(
                b=b, device=output.device, ids=self.pad_tok_id
            )
            # [b, d_vocab]

        ## If not exceed 15 epochs, keep using Teacher Forcing
        if cur_epoch < 15:
            if ith < self.len_ans:
                return self.get_embd_tensor(
                    b, output.device, ans_ids[:, ith].unsqueeze(1)
                )
            return pad_embd
            # [b, d_vocab]

        # Apply Scheduled Sampling
        t = np.random.binomial(1, cur_step / self.max_step)
        if t == 0:
            if ith < self.len_ans:
                return self.get_embd_tensor(
                    b, output.device, ans_ids[:, ith].unsqueeze(1)
                )
            return pad_embd
            # [b, d_vocab]

        #################
        ## Calculate eta
        #################
        r = np.random.binomial(1, max((cur_step / self.max_step, self.bigEPSILON)))
        epsilon = torch.exp(torch.tensor(-self.epsilon))
        if r == 0:
            if ith < self.len_ans:
                output_ = []
                for b_ in range(b):
                    output_.append(output[b_, ans_ids[b_, ith]].view(1))
                output_ = torch.cat(output_, dim=0)
                # [b]
                ita = epsilon * output_
            else:
                ita = epsilon * output[:, self.pad_tok_id]
        else:
            ita = epsilon * torch.max(epsilon * output, dim=-1)[0]
        # ita: [b]

        #################
        ## Calculate mask
        #################
        mask = torch.where(
            output > ita.unsqueeze(1).repeat(1, self.d_vocab),
            torch.ones(output.size()).type_as(output),
            torch.zeros(output.size()).type_as(output),
        )
        # [b, d_vocab]

        #################
        ## Recalculate output
        #################
        output = output * mask
        # [b, d_vocab]
        output = output / torch.sum(output, dim=1).unsqueeze(1).repeat(1, self.d_vocab)
        # [b, d_vocab]

        return output

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
