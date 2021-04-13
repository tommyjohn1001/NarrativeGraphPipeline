'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from collections import defaultdict
import glob, ast, gc, json

from torch.utils.data import Dataset
from torchtext.vocab import Vectors
import torch
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from modules.utils import ParallelHelper
from configs import logging, args, PATH


SEQ_LEN_CONTEXT = args.seq_len_para * args.n_paras

nlp         = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger', "lemmatizer"])
glove_embd  = Vectors("glove.6B.200d.txt", cache=".vector_cache/",
                      url="http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip")

PAD = "[PAD]"
CLS = "[CLS]"
UNK = "[UNK]"
SEP = "[SEP]"

special_toks = set([PAD, CLS, UNK, SEP])

def pad(l, max_len):
    return l + [PAD]*(max_len - len(l)), len(l)

class Vocab:
    def __init__(self, path_vocab) -> None:
        self.stoi   = dict()
        self.itos   = dict()

        # Add special tokens
        for ith, word in enumerate([PAD, CLS, SEP, UNK]):
            self.stoi[word] = ith
            self.itos[ith]  = word

        # COnstruct vocab from token list file
        with open(path_vocab, 'r') as vocab_file:
            for ith, word in enumerate(vocab_file.readlines()):
                word = word.replace('\n', '')
                if word != '':
                    self.stoi[word] = ith
                    self.itos[ith]  = word

    def __len__(self):
        return len(self.stoi)

class CustomDataset(Dataset):
    def __init__(self, path_csv_dir, path_vocab):
        # Search for available data file within directory
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        # Read vocab
        self.vocab  = Vocab(path_vocab)

        self.ques           = None
        self.ques_len       = None
        self.ans1           = None
        self.ans2           = None
        self.ans1_len       = None
        self.ans2_len       = None
        self.ans1_mask      = None
        self.ans2_mask      = None
        self.ans1_tok_idx   = None
        self.ans2_tok_idx   = None
        self.contx          = None
        self.contx_len      = None

        self.n_exchange     = 0


    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        return {
            'ques'          : self.ques[idx],
            'ques_len'      : self.ques_len[idx],
            'contx'         : self.contx[idx],
            'contx_len'     : self.contx_len[idx],
            'ans1'          : self.ans1[idx],
            'ans2'          : self.ans2[idx],
            'ans1_len'      : self.ans1_len[idx],
            'ans2_len'      : self.ans2_len[idx],
            'ans1_mask'     : self.ans1_mask[idx],
            'ans2_mask'     : self.ans2_mask[idx],
            'ans1_tok_idx'  : self.ans1_tok_idx[idx],
            'ans2_tok_idx'  : self.ans2_tok_idx[idx]
        }

    def f_process_file(self, entries, queue, arg):
        for entry in entries.itertuples():

            ###########################
            # Process question
            ###########################
            ques, ques_len  = pad(entry.question.split(' '), args.seq_len_ques)
            ques            = glove_embd.get_vecs_by_tokens(ques).numpy()


            ###########################
            # Process answers including mask token id and embedding form
            ###########################
            answers         = ast.literal_eval(entry.answers)
            ans1, ans2      = answers[0].split(' '), answers[1].split(' ')


            ans1_mask       = np.array([1]*len(ans1) +\
                                       [0]*(args.seq_len_ans - len(ans1)), dtype=np.float)
            ans2_mask       = np.array([1]*len(ans2) +\
                                       [0]*(args.seq_len_ans - len(ans2)), dtype=np.float)

            ans1, ans1_len  = pad(ans1, args.seq_len_ans)
            ans2, ans2_len  = pad(ans2, args.seq_len_ans)

            ans1_tok_idx    = np.array([self.vocab.stoi[w.lower()] if w not in special_toks else self.vocab.stoi[w]
                                        for w in ans1], dtype=np.long)
            ans2_tok_idx    = np.array([self.vocab.stoi[w.lower()] if w not in special_toks else self.vocab.stoi[w] 
                                        for w in ans2], dtype=np.long)

            ans1 = glove_embd.get_vecs_by_tokens(ans1).numpy()
            ans2 = glove_embd.get_vecs_by_tokens(ans2).numpy()


            ###########################
            # Process context
            ###########################
            En      = ast.literal_eval(entry.En)
            Hn      = ast.literal_eval(entry.Hn)

            contx = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]

            # Process context
            contx = ' '.join(contx).split(' ')
            if len(contx) > 2000:
                print(entry.doc_id)

            # Pad context
            contx, contx_len    = pad(contx, SEQ_LEN_CONTEXT)
            # Embed context by GloVe
            contx = glove_embd.get_vecs_by_tokens(contx).numpy()

            # context: [SEQ_LEN_CONTEXT = 1600, d_embd = 200]

            queue.put({
                'ques'          : ques,
                'ques_len'      : ques_len,
                'ans1'          : ans1,
                'ans2'          : ans2,
                'ans1_len'      : ans1_len,
                'ans2_len'      : ans2_len,
                'ans1_mask'     : ans1_mask,
                'ans2_mask'     : ans2_mask,
                'ans1_tok_idx'  : ans1_tok_idx,
                'ans2_tok_idx'  : ans2_tok_idx,
                'contx'         : contx,
                'contx_len'     : contx_len
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        self.ques           = []
        self.ques_len       = []
        self.ans1           = []
        self.ans2           = []
        self.ans1_len       = []
        self.ans2_len       = []
        self.ans1_mask      = []
        self.ans2_mask      = []
        self.ans1_tok_idx   = []
        self.ans2_tok_idx   = []
        self.contx          = []
        self.contx_len      = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        entries = ParallelHelper(self.f_process_file, df, num_proc=args.num_proc,
                                 data_allocation=lambda dat, l, h: dat.iloc[l:h]).launch()
        # with Pool(args.num_proc) as pool:
        #     entries = list(tqdm(pool.imap(f_process_file, zip(df.to_dict(orient='records'), repeat(self.vocab), repeat(self.n_exchange))),
                                # desc="", total=len(df)))

        for entry in entries:
            self.ques.append(entry['ques'])
            self.ques_len.append(entry['ques_len'])
            self.ans1.append(entry['ans1'])
            self.ans2.append(entry['ans2'])
            self.ans1_len.append(entry['ans1_len'])
            self.ans2_len.append(entry['ans2_len'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans2_mask.append(entry['ans2_mask'])
            self.ans1_tok_idx.append(entry['ans1_tok_idx'])
            self.ans2_tok_idx.append(entry['ans2_tok_idx'])
            self.contx.append(entry['contx'])
            self.contx_len.append(entry['contx_len'])

    def switch_answerability(self):
        self.n_exchange += 1



def build_vocab_PGD():
    """ Build vocab for Pointer Generator Decoder. """
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
    with open(PATH['vocab_PGD'], 'w+') as vocab_file:
        for word in words:
            vocab_file.write(word + '\n')
