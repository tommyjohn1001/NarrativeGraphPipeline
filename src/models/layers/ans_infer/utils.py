from transformers.generation_utils import *
import torch.nn as torch_nn
import torch

class CustomGen(GenerationMixin):
    def __init__(self,
        batch_size: int = 2,
        min_length: int = 5,
        max_length: int = 20,
        num_beams: int = 10,
        temperature: int = 1,
        no_repeat_ngram_size: int = 5,
        model: Any = None,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        device: Any = 'cpu') -> None:
        super().__init__()

        self.batch_size     = batch_size
        self.max_length     = max_length
        self.num_beams      = num_beams
        self.pad_token_id   = pad_token_id
        self.bos_token_id   = bos_token_id
        self.eos_token_id   = eos_token_id
        self.device         = device


        self.model  = model

        ############
        ## Declare necessary components

        self.beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            do_early_stopping=False
        )

        self.logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(min_length, eos_token_id=eos_token_id),
            NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)
        ])

        self.stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=None,
        )

        # instantiate logits processors
        self.logits_warper = self._get_logits_warper(
            top_k=50, top_p=1, temperature=temperature, num_beams=num_beams
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs
        }

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        encoder_outputs: Any = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
        """

        # init values
        output_scores       = False
        output_attentions   = False

        output_hidden_states    = False
        return_dict_in_generate = False
        is_encoder_decoder      = False

        if isinstance(model_kwargs, dict):
            model_kwargs["encoder_outputs"] = encoder_outputs
        else:
            model_kwargs = {"encoder_outputs": encoder_outputs}

        # init attention / hidden states / scores tuples
        scores                  = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions      = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions        = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states   = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(self.beam_scorer._beam_hyps)
        num_beams  = self.beam_scorer.num_beams

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = torch.ones((batch_size * num_beams, 1), dtype=torch.long, device=self.device) * self.bos_token_id

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < self.max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=self.max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if is_encoder_decoder else (outputs.attentions,)
                    )
                    if is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = self.beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
            )

            if self.beam_scorer.is_done:
                break

            if self.stopping_criteria(input_ids, scores):
                break

        sequence_outputs = self.beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
