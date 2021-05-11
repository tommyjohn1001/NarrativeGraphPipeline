'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from collections import defaultdict
import glob, ast, gc, json

from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from modules.utils import ParallelHelper
from configs import logging, args, PATH


SEQ_LEN_CONTEXT = args.seq_len_para * args.n_paras


class Vocab:
    def __init__(self, path_vocab=None) -> None:
        if path_vocab is None:
            path_vocab  =PATH['vocab']


        self.dict_stoi   = dict()
        self.dict_itos   = dict()

        self.pad    = "[pad]"
        self.cls    = "[cls]"
        self.sep    = "[sep]"
        self.unk    = "[unk]"
        self.pad_id = 0
        self.cls_id = 1
        self.sep_id = 2
        self.unk_id = 3

        # Construct vocab from token list file
        with open(path_vocab, 'r') as vocab_file:
            for ith, word in enumerate(vocab_file.readlines()):
                word = word.replace('\n', '')
                if word != '':
                    self.dict_stoi[word] = ith
                    self.dict_itos[ith]  = word

    def __len__(self):
        return len(self.dict_stoi)

    def stoi(self, toks):
        def s_to_id(tok):
            try:
                id_ = self.dict_stoi[tok]
            except KeyError:
                id_ = self.dict_stoi[self.unk]

            return id_

        if isinstance(toks, torch.Tensor) or isinstance(toks, np.ndarray):
            toks    = toks.tolist()


        if isinstance(toks, str):
            return s_to_id(toks)
        if isinstance(toks[0], str):
            return list(map(s_to_id, toks))
        elif isinstance(toks[0], list):
            return [list(map(s_to_id, tok)) for tok in toks]
        else:
            raise TypeError(f"'toks' must be 'list' or 'int' type. Got {type(toks)}")

    def itos(self, ids):
        def id_to_s(id_):
            try:
                tok = self.dict_itos[id_]
            except KeyError:
                tok = self.unk

            return tok

        if isinstance(ids, torch.Tensor) or isinstance(ids, np.ndarray):
            ids    = ids.tolist()


        if isinstance(ids, int):
            return id_to_s(ids)
        if isinstance(ids[0], int):
            return list(map(id_to_s, ids))
        elif isinstance(ids[0], list):
            return [list(map(id_to_s, ids_)) for ids_ in ids]
        else:
            raise TypeError(f"'ids' must be 'list' or 'int' type. Got {type(ids)}")


class CustomDataset(Dataset):
    def __init__(self, path_csv_dir, vocab: Vocab):
        # Search for available data file within directory
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        # Read vocab
        self.vocab      = vocab

        self.nlp_bert   = BertTokenizer.from_pretrained(args.bert_model)

        # NOTE: These two fields are for debugging only
        # self.docId          = None
        # self.ques_plain     = None
        self.ques           = None
        self.ques_mask      = None
        self.ans1           = None # in token ID form, not embedded form
        self.ans2           = None # in token ID form, not embedded form
        self.ans1_mask      = None
        self.ans1_loss      = None
        self.paras          = None
        self.paras_mask     = None

        self.n_exchange     = args.n_paras // 2


    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        return {
            # 'docId'         : self.docId[idx],
            # 'ques_plain'    : self.ques_plain[idx],
            'ques'          : self.ques[idx],
            'ques_mask'     : self.ques_mask[idx],
            'ans1'          : self.ans1[idx],
            'ans2'          : self.ans2[idx],
            'ans1_mask'     : self.ans1_mask[idx],
            'ans1_loss'     : self.ans1_loss[idx],
            'paras'         : self.paras[idx],
            'paras_mask'    : self.paras_mask[idx]
        }

    def process_sent(self, sent:str, max_len: int) -> tuple:
        """Process sentence (question, a sentence in context or answer).

        Args:
            sent (str): sentence to be processed
            max_len (int): predefined max len of sent to be padded

        Returns:
            tuple: tuple containing numpy arrays
        """

        sent_       = sent.lower().split(' ')

        sent_       = self.nlp_bert.convert_tokens_to_ids(sent_)

        cls_tok_id  = self.nlp_bert.cls_token_id
        sep_tok_id  = self.nlp_bert.sep_token_id
        pad_tok_id  = self.nlp_bert.pad_token_id

        sent_       = [cls_tok_id] + sent_[:max_len-2] + [sep_tok_id]

        sent_len_   = len(sent_)
        sent_mask_  = np.array([1]*sent_len_ + [0]*(max_len - sent_len_), dtype=np.float)
        sent_       = np.array(sent_ + [pad_tok_id]*(max_len - sent_len_), dtype=np.int)

        return sent_, np.array(sent_len_), sent_mask_

    def process_ans_loss(self, sent:str, max_len: int):
        sent_       = sent.lower().split(' ')

        sent_       = self.vocab.stoi(sent_)

        pad_tok_id  = self.nlp_bert.pad_token_id

        sent_len_   = len(sent_)
        sent_       = np.array(sent_ + [pad_tok_id]*(max_len - sent_len_), dtype=np.int)

        return sent_

    def f_process_file(self, entries, queue, arg):
        for entry in entries.itertuples():
            ###########################
            # Process question
            ###########################
            ques, _, ques_mask  = self.process_sent(entry.question, args.seq_len_ques)


            ###########################
            # Process answers
            ###########################
            answers         = ast.literal_eval(entry.answers)

            # This trick ensures training process occurs in longer answer
            if len(' '.split(answers[0])) < len(' '.split(answers[1])):
                answers[0], answers[1] = answers[1], answers[0]

            ans1, _, ans1_mask  = self.process_sent(answers[0], args.seq_len_ans)
            ans2, _, _          = self.process_sent(answers[1], args.seq_len_ans)
            ans1_loss           = self.process_ans_loss(answers[0], args.seq_len_ans)


            ###########################
            # Process context
            ###########################
            En      = ast.literal_eval(entry.En)
            Hn      = ast.literal_eval(entry.Hn)

            contx = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]

            # Process context
            paras, paras_mask = [], []
            for sent in contx:
                sent, _, sent_mask = self.process_sent(sent, args.seq_len_para)
                paras.append(np.expand_dims(sent, axis=0))
                paras_mask.append(np.expand_dims(sent_mask, axis=0))

            paras       = np.vstack(paras)
            paras_mask  = np.vstack(paras_mask)


            queue.put({
                'ques'          : ques,
                'ques_mask'     : ques_mask,
                'ans1'          : ans1,
                'ans2'          : ans2,
                'ans1_mask'     : ans1_mask,
                'ans1_loss'     : ans1_loss,
                'paras'         : paras,
                'paras_mask'    : paras_mask
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        # self.docId          = []
        # self.ques_plain     = []
        self.ques           = []
        self.ques_mask      = []
        self.ans1           = []
        self.ans2           = []
        self.ans1_mask      = []
        self.ans1_loss      = []
        self.paras          = []
        self.paras_mask     = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        entries = ParallelHelper(self.f_process_file, df, lambda dat, l, h: dat.iloc[l:h],
                                 args.num_proc).launch()

        for entry in entries:
            # self.docId.append(entry['docId'])
            # self.ques_plain.append(entry['ques_plain'])
            self.ques.append(entry['ques'])
            self.ques_mask.append(entry['ques_mask'])
            self.ans1.append(entry['ans1'])
            self.ans2.append(entry['ans2'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans1_loss.append(entry['ans1_loss'])
            self.paras.append(entry['paras'])
            self.paras_mask.append(entry['paras_mask'])

    def switch_answerability(self):
        if self.n_exchange == args.n_paras:
            self.n_exchange = 0
        else:
            self.n_exchange += args.n_paras // 2



def build_vocab():
    """ Build vocab for Inferring answer module. """
    log = logging.getLogger("spacy")
    log.setLevel(logging.ERROR)

    nlp_            = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])
    nlp_.max_length  = 2500000


    word_count      = defaultdict(int)

    answer_toks     = set()


    # Read stories in train, test, valid set
    for split in ["train", "test", "validation"]:
        logging.info(f"Process split: {split}")

        path    = PATH['processed_contx'].replace("[SPLIT]", split)
        with open(path, 'r') as d_file:
            contexts    = json.load(d_file)

        # Add tokens in context to global vocab
        for context in tqdm(contexts.values(), desc="Get context"):
            for para in context:
                for tok in nlp_(para):
                    if  not tok.is_punct and\
                        not tok.is_stop and\
                        not tok.like_url:
                        word_count[tok.text] += 1

        # Add tokens in ans1 and ans2 to global vocab
        dataset = load_dataset("narrativeqa", split=split)
        for entry in tqdm(dataset, total=len(dataset), desc="Get answers"):
            for tok in entry['answers'][0]['tokens'] + entry['answers'][1]['tokens']:
                answer_toks.add(tok.lower())


    # Sort word_count dict and filter top 1000 words
    words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    words = set(word for word, occurence in words[:1000] if occurence >= args.min_count_PGD)

    # words = set(w for w, occurence in word_count.items() if occurence >= args.min_count_PGD)

    words = words.union(answer_toks)

    # Write vocab to TXT file
    with open(PATH['vocab'], 'w+') as vocab_file:
        for word in ["[pad]", "[cls]", "[sep]", "[unk]"]:
            vocab_file.write(word + '\n')

    # Write vocab to TXT file
    with open(PATH['vocab'], 'w+') as vocab_file:
        for word in words:
            vocab_file.write(word + '\n')
