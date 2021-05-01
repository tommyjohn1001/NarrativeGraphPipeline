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


class Vocab:
    def __init__(self, path_vocab=None) -> None:
        if path_vocab is None:
            path_vocab  =PATH['vocab_PGD']

        self.glove_embd = Vectors("glove.6B.200d.txt", cache=".vector_cache/")

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
        self.pad_vec= np.full((200,), 0)
        self.cls_vec= np.full((200,), 1)
        self.sep_vec= np.full((200,), 2)
        self.unk_vec= np.full((200,), 3)

        # Construct vocab from token list file
        with open(path_vocab, 'r') as vocab_file:
            for ith, word in enumerate(vocab_file.readlines()):
                word = word.replace('\n', '')
                if word != '':
                    self.dict_stoi[word] = ith
                    self.dict_itos[ith]  = word

    def __len__(self):
        return len(self.dict_stoi)

    def stoi(self, tok):
        try:
            id_ = self.dict_stoi[tok]
        except KeyError:
            id_ = self.dict_stoi[self.unk]

        return id_

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

    def convert_tokens_to_ids(self, toks: list):
        return list(map(self.stoi, toks))

    def get_vecs_by_toks(self, toks):
        return self.glove_embd.get_vecs_by_tokens(toks).numpy()

    def get_vecs_by_tokids(self, toks):
        if isinstance(toks, torch.Tensor) or isinstance(toks, np.ndarray):
            toks    = toks.tolist()

        def tokid_to_vec(tok_id):
            if tok_id == self.pad_id:
                return self.pad_vec
            elif tok_id == self.cls_id:
                return self.cls_vec
            elif tok_id == self.sep_id:
                return self.sep_vec
            elif tok_id == self.unk_id:
                return self.unk_vec
            else:
                return self.glove_embd.get_vecs_by_tokens(self.itos(tok_id)).numpy()

        if isinstance(toks, int):
            return tokid_to_vec(toks)
        if isinstance(toks[0], int):
            return np.array(list(map(tokid_to_vec, toks)))
        elif isinstance(toks[0], list):
            vecs    = [list(map(tokid_to_vec, toks_)) for toks_ in toks]

            return np.array(vecs)
        else:
            raise TypeError(f"'toks' must be 'list' or 'int' type. Got {type(toks)}")
            

    def padding(self, l, max_len):
        return l + [self.pad]*(max_len - len(l)), len(l)



class CustomDataset(Dataset):
    def __init__(self, path_csv_dir, vocab: Vocab):
        # Search for available data file within directory
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        # Read vocab
        self.vocab  = vocab

        self.docId          = None
        self.ques_plain     = None
        self.ques           = None
        self.ques_len       = None
        self.ans1           = None
        self.ans1_len       = None
        self.ans1_mask      = None
        self.ans1_tok_idx   = None
        self.ans2_tok_idx   = None
        self.contx          = None
        self.contx_len      = None

        self.n_exchange     = args.n_paras // 2


    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        return {
            'docId'         : self.docId[idx],
            'ques_plain'    : self.ques_plain[idx],
            'ques'          : self.ques[idx],
            'ques_len'      : self.ques_len[idx],
            'contx'         : self.contx[idx],
            'contx_len'     : self.contx_len[idx],
            'ans1'          : self.ans1[idx],
            'ans1_len'      : self.ans1_len[idx],
            'ans1_mask'     : self.ans1_mask[idx],
            'ans1_tok_idx'  : self.ans1_tok_idx[idx],
            'ans2_tok_idx'  : self.ans2_tok_idx[idx]
        }

    def f_process_file(self, entries, queue, arg):
        vocab: Vocab    = arg[0]
        for entry in entries.itertuples():

            ###########################
            # Process question
            ###########################
            ques, ques_len  = vocab.padding(entry.question.split(' '), args.seq_len_ques)
            ques            = vocab.get_vecs_by_toks(ques)


            ###########################
            # Process answers including mask token id and embedding form
            ###########################
            answers         = ast.literal_eval(entry.answers)
            ans1, ans2      = answers[0].split(' '), answers[1].split(' ')

            if len(ans1) < len(ans2):
                ans1, ans2 = ans2, ans1

            ans1            = [vocab.cls] + ans1 + [vocab.sep]
            ans2            = [vocab.cls] + ans2 + [vocab.sep]

            ans1_mask       = np.array([1]*len(ans1) +\
                                       [0]*(args.seq_len_ans - len(ans1)), dtype=np.float)

            ans1, ans1_len  = vocab.padding(ans1, args.seq_len_ans)
            ans2, _         = vocab.padding(ans2, args.seq_len_ans)

            ans1_tok_idx    = np.array([vocab.stoi(w.lower()) for w in ans1], dtype=np.long)
            ans2_tok_idx    = np.array([vocab.stoi(w.lower()) for w in ans2], dtype=np.long)

            ans1            = vocab.get_vecs_by_toks(ans1)


            ###########################
            # Process context
            ###########################
            En      = ast.literal_eval(entry.En)
            Hn      = ast.literal_eval(entry.Hn)

            contx = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]

            # Process context
            contx = ' '.join(contx).split(' ')

            # Pad context
            contx, contx_len    = vocab.padding(contx, SEQ_LEN_CONTEXT)
            # Embed context by GloVe
            contx = vocab.get_vecs_by_toks(contx)

            # context: [SEQ_LEN_CONTEXT = 1600, d_embd = 200]

            queue.put({
                'docId'         : entry.doc_id,
                'ques_plain'    : entry.question,
                'ques'          : ques,
                'ques_len'      : ques_len,
                'ans1'          : ans1,
                'ans1_len'      : ans1_len,
                'ans1_mask'     : ans1_mask,
                'ans1_tok_idx'  : ans1_tok_idx,
                'ans2_tok_idx'  : ans2_tok_idx,
                'contx'         : contx,
                'contx_len'     : contx_len
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        self.docId          = []
        self.ques_plain     = []
        self.ques           = []
        self.ques_len       = []
        self.ans1           = []
        self.ans1_len       = []
        self.ans1_mask      = []
        self.ans1_tok_idx   = []
        self.ans2_tok_idx   = []
        self.contx          = []
        self.contx_len      = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        entries = ParallelHelper(self.f_process_file, df, lambda dat, l, h: dat.iloc[l:h],
                                 args.num_proc, self.vocab).launch()
        # with Pool(args.num_proc) as pool:
        #     entries = list(tqdm(pool.imap(f_process_file, zip(df.to_dict(orient='records'), repeat(self.vocab), repeat(self.n_exchange))),
                                # desc="", total=len(df)))

        for entry in entries:
            self.docId.append(entry['docId'])
            self.ques_plain.append(entry['ques_plain'])
            self.ques.append(entry['ques'])
            self.ques_len.append(entry['ques_len'])
            self.ans1.append(entry['ans1'])
            self.ans1_len.append(entry['ans1_len'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans1_tok_idx.append(entry['ans1_tok_idx'])
            self.ans2_tok_idx.append(entry['ans2_tok_idx'])
            self.contx.append(entry['contx'])
            self.contx_len.append(entry['contx_len'])

    def switch_answerability(self):
        if self.n_exchange == args.n_paras:
            self.n_exchange == 0
        else:
            self.n_exchange += args.n_paras // 2



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
