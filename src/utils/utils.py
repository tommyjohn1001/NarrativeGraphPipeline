import logging as log_std
import warnings
from typing import List, Sequence


import pytorch_lightning as pl
import rich.syntax
import rich.tree
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=log_std.INFO) -> log_std.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = log_std.getLogger(name)
    logger.setLevel(level)

    # this ensures all log_std levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable log_std any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from log_std hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()


######################################################################################
# User-defined utils
######################################################################################
import re, multiprocessing

from transformers.generation_utils import *
import torch
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


class ParallelHelper:
    def __init__(self, f_task: object, data: list,
                 data_allocation: object, num_workers: int = 4, desc=None):
        self.n_data = len(data)

        self.queue  = multiprocessing.Queue()
        # self.pbar   = tqdm(total=self.n_data, desc=desc)

        self.jobs = list()
        for ith in range(num_workers):
            lo_bound = ith * self.n_data // num_workers
            hi_bound = (ith + 1) * self.n_data // num_workers \
                if ith < (num_workers - 1) else self.n_data

            p = multiprocessing.Process(target=f_task,
                                        args=(data_allocation(data, lo_bound, hi_bound),
                                              self.queue))
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task

        """
        dataset = []

        for job in self.jobs:
            job.start()

        cnt = 0
        while cnt < self.n_data:
            while not self.queue.empty():
                dataset.append(self.queue.get())
                cnt += 1

                # self.pbar.update()

        # self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()


        return dataset


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
        eos_token_id: Optional[int] = 2) -> None:
        super().__init__()

        self.batch_size     = batch_size
        self.max_length     = max_length
        self.num_beams      = num_beams
        self.pad_token_id   = pad_token_id
        self.bos_token_id   = bos_token_id
        self.eos_token_id   = eos_token_id


        self.model  = model

        ############
        ## Declare necessary components

        self.beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
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
            input_ids = torch.ones((batch_size * num_beams, 1), dtype=torch.long) * self.bos_token_id

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float)
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


def process_sent(sent: str):
    return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", '', sent).strip()

# ref = [
#     "the transcript is a written version of each day",
#     "version of each day"
# ]
# pred= "a written version of"
EPSILON = 10e-10
def get_scores(ref: list, pred: str):
    """Calculate metrics BLEU-1, BLEU4, METEOR and ROUGE_L.

    Args:
        ref (list): list of reference strings
        pred (str): string generated by model

    Returns:
        tuple: tuple of 4 scores
    """
    pred    = process_sent(pred)
    ref     = list(map(process_sent, ref))

    # Calculate BLEU score
    ref_    = [x.split() for x in ref]
    pred_   = pred.split()

    bleu_1  = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
    bleu_4  = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate METEOR
    meteor  = meteor_score(ref, pred)

    # Calculate ROUGE-L
    scores  = np.array([
        Rouge().get_scores(ref_, pred, avg=True)['rouge-l']['f']
        for ref_ in ref
    ])
    rouge_l = np.mean(scores)



    return bleu_1 if bleu_1 > EPSILON else 0,\
        bleu_4 if bleu_4 > EPSILON else 0,\
        meteor if meteor > EPSILON else 0,\
        rouge_l if rouge_l > EPSILON else 0