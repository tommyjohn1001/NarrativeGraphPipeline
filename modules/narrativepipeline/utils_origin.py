'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from collections import defaultdict
import glob, ast, gc

from torch.utils.data import Dataset
from torchtext.vocab import Vectors
import torch
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from modules.data_reading.data_reading import clean_end, clean_text
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
    return l + [PAD]*(max_len - len(l))

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
        self.ans1           = None
        self.ans2           = None
        self.ans1_mask      = None
        self.ans2_mask      = None
        self.ans1_tok_idx   = None
        self.ans2_tok_idx   = None
        self.contx          = None

        self.n_exchange     = 0


    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'ques'          : self.ques[idx],
            'contx'         : self.contx[idx],
            'ans1'          : self.ans1[idx],
            'ans2'          : self.ans2[idx],
            'ans1_mask'     : self.ans1_mask[idx],
            'ans2_mask'     : self.ans2_mask[idx],
            'ans1_tok_idx'  : self.ans1_tok_idx[idx],
            'ans2_tok_idx'  : self.ans2_tok_idx[idx]
            # 'ans1_text'     : self.ans1_text[idx]
        }

    def f_process_file(self, entries, queue):
        for entry in entries.itertuples():

            ###########################
            # Process question
            ###########################
            question    = pad(entry.question.split(' '), args.seq_len_ques)
            question    = glove_embd.get_vecs_by_tokens(question).numpy()


            ###########################
            # Process answers including mask token id and embedding form
            ###########################
            answers     = ast.literal_eval(entry.answers)
            answer1     = answers[0].split(' ')
            answer2     = answers[1].split(' ')


            answer1_mask    = np.array([1]*len(answer1) +\
                                [0]*(args.seq_len_ans - len(answer1)), dtype=np.float)
            answer2_mask    = np.array([1]*len(answer2) +\
                                [0]*(args.seq_len_ans - len(answer2)), dtype=np.float)

            answer1         = pad(answer1, args.seq_len_ans)
            answer2         = pad(answer2, args.seq_len_ans)

            answer1_tok_idx = np.array([self.vocab.stoi[w.lower()] if w not in special_toks else self.vocab.stoi[w]
                                for w in answer1], dtype=np.long)
            answer2_tok_idx = np.array([self.vocab.stoi[w.lower()] if w not in special_toks else self.vocab.stoi[w]
                                for w in answer2], dtype=np.long)

            answer1 = glove_embd.get_vecs_by_tokens(answer1).numpy()
            answer2 = glove_embd.get_vecs_by_tokens(answer2).numpy()


            ###########################
            # Process context
            ###########################
            En      = ast.literal_eval(entry.En)
            Hn      = ast.literal_eval(entry.Hn)

            context = En[self.n_exchange:] + Hn[:self.n_exchange]

            # Process context

            # Remove HTML tag
            context = BeautifulSoup(' '.join(context), 'html.parser').get_text()
            # Tokenize context
            context = [tok.text for tok in nlp(context)]

            # Pad context
            context = pad(context, SEQ_LEN_CONTEXT)
            # Embed context by GloVe
            context = glove_embd.get_vecs_by_tokens(context).numpy()

            # context: [SEQ_LEN_CONTEXT = 1600, d_embd = 200]

            queue.put({
                'ques'          : question,
                'ans1'          : answer1,
                'ans2'          : answer2,
                'ans1_mask'     : answer1_mask,
                'ans2_mask'     : answer2_mask,
                'ans1_tok_idx'  : answer1_tok_idx,
                'ans2_tok_idx'  : answer2_tok_idx,
                'contx'         : context
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        self.ques           = []
        self.ans1           = []
        self.ans2           = []
        self.ans1_mask      = []
        self.ans2_mask      = []
        self.ans1_tok_idx   = []
        self.ans2_tok_idx   = []
        self.contx          = []

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
            self.ans1.append(entry['ans1'])
            self.ans2.append(entry['ans2'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans2_mask.append(entry['ans2_mask'])
            self.ans1_tok_idx.append(entry['ans1_tok_idx'])
            self.ans2_tok_idx.append(entry['ans2_tok_idx'])
            self.contx.append(entry['contx'])

    def switch_answerability(self):
        self.n_exchange += 1



def build_vocab_PGD():
    """ Build vocab for Pointer Generator Decoder. """
    log = logging.getLogger("spacy")
    log.setLevel(logging.ERROR)

    nlp_            = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])
    nlp_.max_length  = 2500000

    read_stories    = set()
    word_count      = defaultdict(int)

    answer_toks     = set()


    # Read stories in train, test, valid set
    for split in ["train", "test", "validation"]:
        dataset = load_dataset("narrativeqa", split=split)

        for entry in tqdm(dataset, desc=f"Split '{split}'"):

            # Add tokens in context to global vocab
            if entry['document']['id'] not in read_stories:     # This line ensures we dont read a document twice

                read_stories.add(entry['document']['id'])

                # Process text
                context = entry['document']['text']

                ## Extract text from HTML
                soup    = BeautifulSoup(context, 'html.parser')
                if soup.pre is not None:
                    context = ''.join(list(soup.pre.findAll(text=True)))

                ## Clean and lowercase
                context = clean_text(context)


                start_  = entry['document']['start'].lower()
                end_    = entry['document']['end'].lower()
                end_    = clean_end(end_)

                start_  = context.find(start_)
                end_    = context.rfind(end_)
                if start_ == -1:
                    start_ = 0
                if end_ == -1:
                    end_ = len(context)

                context = context[start_:end_]

                for tok in nlp_(context):
                    if  not tok.is_punct and\
                        not tok.is_stop and\
                        not tok.like_url:
                        word_count[tok.text] += 1

            # Add tokens in answer1 and answer2 to global vocab
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
