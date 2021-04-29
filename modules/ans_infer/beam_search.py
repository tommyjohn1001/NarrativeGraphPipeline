
from typing import Optional

import torch.nn.functional as torch_f
import torch
from transformers.generation_utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    validate_stopping_criteria,
    GenerationMixin,
    MaxLengthCriteria,
    BeamSearchScorer
)

from modules.narrativepipeline.utils import Vocab
from configs import args

class BeamSearch(GenerationMixin):
    def __init__(self, vocab: Vocab, model: torch.nn.Module ) -> None:
        self.vocab  = vocab
        self.model  = model

    def beam_search(self,
        batch_size: int,
        num_beams: int,
        model_args: tuple,
        max_length: int,
        length_penalty: Optional[float] = 1,
        early_stopping: Optional[bool] = False,
        no_repeat_ngram_size: Optional[int] = 5,
        input_ids: Optional[torch.Tensor] = None,
        **model_kwargs)->torch.LongTensor:

        pad_token_id = self.vocab.pad_id
        eos_token_id = self.vocab.sep_id
        bos_token_id = self.vocab.cls_id

        logits_processor    = LogitsProcessorList(
            repetition_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=max_length,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
        )
        stopping_criteria   = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        beam_scorer         = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=args.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=1,
        )
        validate_stopping_criteria(stopping_criteria, max_length)


        # init scores
        scores       = None

        # init CLS as input ids
        if input_ids is None:
            input_ids   = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=args.device)
        input_ids   = input_ids.repeat((num_beams, 1))
        # batch_beam = batch * n_beams, 1


        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        # Init beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=args.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:

            # Feed into model

            output = self.model(input_ids[:, -1], *model_args)
            next_token_logits = torch.logit(output)[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `torch_flog_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = torch_f.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            
            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores      = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx         = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )


        return sequence_outputs["sequences"]
