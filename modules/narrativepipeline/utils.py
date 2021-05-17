'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from collections import defaultdict
from itertools import combinations
import glob, ast, gc, json

from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from modules.utils import ParallelHelper
from configs import logging, args, PATH


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
    def __init__(self, path_csv_dir):
        # Search for available data file within directory
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        self.nlp_spacy  = spacy.load("en_core_web_sm")
        self.nlp_bert   = BertTokenizer.from_pretrained(PATH['bert'])

        self.vocab      = Vocab(PATH['vocab'])


        self.ques           = None
        self.ques_mask      = None
        self.ans1_bert_ids  = None  # in token ID form, not embedded form
        self.ans1_bert_mask = None
        self.ans1_vocab_ids = None
        self.ans2_vocab_ids = None
        self.paras          = None
        self.paras_len      = None  # this is no. nodes (paras)
        self.paras_mask     = None
        self.edge_indx      = None
        self.edge_len       = None

        self.n_exchange     = 0


    ###########################################
    # OVERIDDEN METHOD
    ###########################################
    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'ques'          : self.ques[idx],
            'ques_mask'     : self.ques_mask[idx],
            'ans1_bert_ids' : self.ans1_bert_ids[idx],
            'ans1_bert_mask': self.ans1_bert_mask[idx],
            'ans1_vocab_ids': self.ans1_vocab_ids[idx],
            'ans2_vocab_ids': self.ans2_vocab_ids[idx],
            'paras'         : self.paras[idx],
            'paras_len'     : self.paras_len[idx],
            'paras_mask'    : self.paras_mask[idx],
            'edge_indx'     : self.edge_indx[idx],
            'edge_len'      : self.edge_len[idx]
        }


    ###########################################
    # USER-DEFINED METHOD
    ###########################################
    def process_sent(self, sent:str, max_len: int, nlp: str = 'bert') -> tuple:
        """Process sentence (question, a sentence in context or answer).

        Args:
            sent (str): sentence to be processed
            max_len (int): predefined max len of sent to be padded

        Returns:
            tuple: tuple containing numpy arrays
        """

        sent_       = sent.lower().split(' ')

        if nlp == "bert":
            sent_       = self.nlp_bert.convert_tokens_to_ids(sent_)

            cls_tok_id  = self.nlp_bert.cls_token_id
            sep_tok_id  = self.nlp_bert.sep_token_id
            pad_tok_id  = self.nlp_bert.pad_token_id
        elif nlp == 'vocab':
            sent_       = self.vocab.stoi(sent_)

            cls_tok_id  = self.vocab.cls_id
            sep_tok_id  = self.vocab.sep_id
            pad_tok_id  = self.vocab.pad_id
        else:
            raise TypeError("'nlp' not correct.")

        sent_       = [cls_tok_id] + sent_ + [sep_tok_id]

        sent_len_   = len(sent_)
        sent_mask_  = np.array([1]*sent_len_ + [0]*(max_len - sent_len_), dtype=np.float)
        sent_       = np.array(sent_ + [pad_tok_id]*(max_len - sent_len_), dtype=np.int)

        return sent_, np.array(sent_len_), sent_mask_

    def construct_edge_indx(self, paras: list, question: str) -> tuple:
        """Construct graph edges' index using raw question and paras

        Args:
            paras (list): list of raw para
            question (str): raw question

        Returns:
            tuple:
                numpy.array: array containing graph's edges
                int: edge len
        """

        paras.insert(0, question)

        para_vocab  = defaultdict(set)

        #####################
        # Construct para vocab
        #####################
        for ith, para in enumerate(paras):
            for tok in self.nlp_spacy(para):
                if not (tok.is_stop or tok.is_punct):
                    para_vocab[tok.text].add(ith)

        #####################
        # Establish edges from para vocab
        #####################
        edges = set()
        for list_paras in para_vocab.values():
            for pair in combinations(list_paras, 2):
                s, d = pair
                edges.add(f"{s}-{d}")
                edges.add(f"{d}-{s}")

        vertex_s, vertex_d = [], []
        for edge in edges:
            s, d = edge.split('-')
            vertex_s.append(int(s))
            vertex_d.append(int(d))

        edge_index  = np.array([vertex_s, vertex_d])
        # [2, *]
        edge_len    = edge_index.shape[1]

        #####################
        # Pad edge index to n_edges
        #####################
        pad         = np.zeros((2, args.n_edges - edge_index.shape[1]))
        edge_index  = np.concatenate((edge_index, pad), axis=1)
        # [2, n_edges]

        return edge_index, edge_len

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
            ans1, ans2      = answers[0].split(' '), answers[1].split(' ')

            # This trick ensures training process occurs in longer answer
            if len(ans1) < len(ans2):
                answers[0], answers[1] = answers[1], answers[0]

            ans1_bert_ids, _, ans1_bert_mask    = self.process_sent(answers[0], args.seq_len_ans)
            ans1_vocab_ids, _, _                = self.process_sent(answers[0], args.seq_len_ans, 'vocab')
            ans2_vocab_ids, _, _                = self.process_sent(answers[1], args.seq_len_ans, 'vocab')


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

            paras_len   = np.array(len(paras))
            paras       = np.vstack(paras)
            paras_mask  = np.vstack(paras_mask)

            # Pad paras and paras_mask
            if paras.shape[0] < args.n_paras:
                print(f"Occurs: {paras.shape[0]}")
                print(entry.doc_id)
                print(entry.question)
                pad = np.zeros((args.n_paras - paras.shape[0], args.seq_len_para))
                paras       = np.concatenate((paras, pad), axis=0) 
                paras_mask  = np.concatenate((paras_mask, pad), axis=0)


            ###########################
            # Construct edges of graph
            ###########################
            edge_indx, edge_len = self.construct_edge_indx(contx, entry.question)


            queue.put({
                'ques'          : ques,
                'ques_mask'     : ques_mask,
                'ans1_bert_ids' : ans1_bert_ids,
                'ans1_bert_mask': ans1_bert_mask,
                'ans1_vocab_ids': ans1_vocab_ids,
                'ans2_vocab_ids': ans2_vocab_ids,
                'paras'         : paras,
                'paras_len'     : paras_len,
                'paras_mask'    : paras_mask,
                'edge_indx'     : edge_indx,
                'edge_len'      : edge_len
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)


        self.ques           = []
        self.ques_mask      = []
        self.ans1_bert_ids  = []
        self.ans1_bert_mask = []
        self.ans1_vocab_ids = []
        self.ans2_vocab_ids = []
        self.paras          = []
        self.paras_len      = []
        self.paras_mask     = []
        self.edge_indx      = []
        self.edge_len       = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        entries = ParallelHelper(self.f_process_file, df, lambda dat, l, h: dat.iloc[l:h],
                                 args.num_proc).launch()

        for entry in entries:
            self.ques.append(entry['ques'])
            self.ques_mask.append(entry['ques_mask'])
            self.ans1_bert_ids.append(entry['ans1_bert_ids'])
            self.ans1_bert_mask.append(entry['ans1_bert_mask'])
            self.ans1_vocab_ids.append(entry['ans1_vocab_ids'])
            self.ans2_vocab_ids.append(entry['ans2_vocab_ids'])
            self.paras.append(entry['paras'])
            self.paras_len.append(entry['paras_len'])
            self.paras_mask.append(entry['paras_mask'])
            self.edge_indx.append(entry['edge_indx'])
            self.edge_len.append(entry['edge_len'])

    def switch_answerability(self):
        self.n_exchange += 1



def build_vocab():
    """ Build vocab for Inferring answer module. """
    log = logging.getLogger("spacy")
    log.setLevel(logging.ERROR)

    nlp_            = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])


    word_count      = defaultdict(int)

    ques_answer_toks= set()


    # Read processed contexts
    paths    = glob.glob(PATH['processed_contx'].replace("[ID]", "*"))
    for path in tqdm(paths, desc="Get context"):
        with open(path, 'r') as d_file:
            context    = json.load(d_file)

            # Add tokens in context to global vocab
            for para in context:
                for tok in nlp_(para):
                    if  not tok.is_punct and\
                        not tok.is_stop and\
                        not tok.like_url:
                        word_count[tok.text] += 1

    # Add tokens in ans1 ans2, question to global vocab
    documents   = pd.read_csv(f"{PATH['raw_data_dir']}/qaps.csv", header=0, index_col=None)
    for entry in tqdm(documents.itertuples(), total=len(documents), desc="Get question and answers"):
        ques    = entry.question_tokenized.lower().split(' ')
        ans1    = entry.answer1_tokenized.lower().split(' ')
        ans2    = entry.answer2_tokenized.lower().split(' ')

        for tok in ques + ans1 + ans2:
            ques_answer_toks.add(tok)

    # Sort word_count dict and filter top 1000 words
    words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    words = set(word for word, occurence in words[:1000] if occurence >= args.min_count_PGD)

    # words = set(w for w, occurence in word_count.items() if occurence >= args.min_count_PGD)

    words = words.union(ques_answer_toks)

    # Write vocab to TXT file
    with open(PATH['vocab'], 'w+') as vocab_file:
        for word in ["[pad]", "[cls]", "[sep]", "[unk]"]:
            vocab_file.write(word + '\n')

    # Write vocab to TXT file
    with open(PATH['vocab'], 'w+') as vocab_file:
        for word in words:
            vocab_file.write(word + '\n')
