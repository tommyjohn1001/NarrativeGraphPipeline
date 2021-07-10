from typing import Any


from transformers.models.bert.tokenization_bert import BertTokenizer
import torch.nn as torch_nn
import torch
import numpy as np

from src.models.layers.reasoning_layer.memory_layer import MemoryBasedReasoning
from src.models.layers.bertbasedembd_layer import BertBasedEmbedding


class Decoder(torch_nn.Module):
    def __init__(
        self,
        len_ans: int = 15,
        d_vocab: int = 30522,
        d_hid: int = 64,
        tokenizer: BertTokenizer = None,
        embd_layer: BertBasedEmbedding = None,
        reasoning: MemoryBasedReasoning = None,
    ):
        super().__init__()

        self.len_ans = len_ans
        self.d_hid = d_hid
        self.tokenizer = tokenizer
        self.d_vocab = d_vocab

        self.t = 0

        self.embd_layer: BertBasedEmbedding = embd_layer
        self.reasoning = reasoning
        self.lin1 = torch_nn.Linear(d_hid, d_vocab)

    def forward(
        self,
        ques: torch.Tensor,
        context: torch.Tensor,
        input_ids: torch.Tensor,
        input_masks: torch.Tensor,
    ):
        # ques: [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]
        # input_ids: [b, len_,]
        # input_masks: [b, len_]

        ans = self.embd_layer.encode_ans(input_ids=input_ids, input_masks=input_masks)
        # [b, len_, d_hid]

        output = self.reasoning(ques=ques, context=context, ans=ans)
        # [b, len_, d_hid]

        output = self.lin1(output)
        # [b, len_, d_vocab]

        return output

    def do_train(
        self,
        ques: torch.Tensor,
        context: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
    ):
        # ques: [b, len_ques, d_hid]
        # context: [b, n_paras, len_para, d_hid]
        # ans_ids   : [b, len_ans]
        # ans_mask  : [b, len_ans]

        b = ques.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (b, 1),
            fill_value=self.tokenizer.cls_token_id,
            device=ques.device,
            requires_grad=False,
        )
        # [b, 1]

        input_ids = [cls_ids]
        outputs = []

        for ith in range(1, self.len_ans):
            output = self(
                ques=ques,
                context=context,
                input_ids=torch.cat(input_ids, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            outputs.append(output[:, -1].unsqueeze(1))

            if ith < self.len_ans - 1:
                ## Apply WEAM
                new_word = self.choose_scheduled_sampling(
                    output=output[:, -1],
                    ans_ids=ans_ids[:, ith],
                    cur_step=cur_step,
                    max_step=max_step,
                    cur_epoch=cur_epoch,
                )
                # [b, 1]

                input_ids.append(new_word)

        output = torch.cat(outputs, dim=1)
        # [b, len_ans - 1, d_vocab]

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output, dim=-1))
        # [b, len_ans - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output_mle, output_ot

    ##############################################
    # Methods for scheduled sampling
    ##############################################

    def choose_scheduled_sampling(
        self,
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        cur_step: int,
        max_step: int,
        cur_epoch: int,
    ):
        # output: [b, d_vocab]
        # ans_ids: [b]

        # Apply Scheduled Sampling
        if cur_epoch < 20:
            t = np.random.binomial(1, min((cur_step / max_step, 0.5)))
        elif cur_epoch < 40:
            t = np.random.binomial(1, min((cur_step / max_step, 0.75)))
        else:
            t = np.random.binomial(1, cur_step / max_step)

        self.t = t

        new_word = (
            ans_ids.unsqueeze(-1)
            if self.t == 0
            else torch.topk(torch.softmax(output, dim=-1), k=1)[1]
        )
        # [b, 1]

        return new_word

    ##############################################
    # Methods for validation/prediction
    ##############################################
    def do_predict(
        self,
        ques: torch.Tensor,
        context: torch.Tensor,
        ans_mask: torch.Tensor,
    ):

        b = ques.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (b, 1),
            fill_value=self.tokenizer.cls_token_id,
            device=ques.device,
            requires_grad=False,
        )
        # [b, 1]

        input_ids = [cls_ids]
        outputs = []

        for ith in range(1, self.len_ans):
            output = self(
                ques=ques,
                context=context,
                input_ids=torch.cat(input_ids, dim=1),
                input_masks=ans_mask[:, :ith],
            )
            # [b, ith, d_vocab]

            outputs.append(output[:, -1].unsqueeze(1))

            if ith < self.len_ans - 1:
                ## Apply WEAM
                new_word = torch.topk(torch.softmax(output[:, -1], dim=-1), k=1)[1]
                # [b, 1]

                input_ids.append(new_word)

        output = torch.cat(outputs, dim=1)
        # [b, len_ans - 1, d_vocab]

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output, dim=-1))
        # [b, len_ans - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, len_ans - 1]

        return output_mle, output_ot
