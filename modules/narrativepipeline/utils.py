'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from typing import Tuple
import glob, ast, gc

from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

from modules.utils import ParallelHelper
from configs import args

bert_tokenizer  = BertTokenizer.from_pretrained(args.bert_model)
cls_id  = bert_tokenizer.cls_token_id
sep_id  = bert_tokenizer.sep_token_id
pad_id  = bert_tokenizer.pad_token_id


class CustomDataset(Dataset):
    def __init__(self, path_csv_dir):
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        self.ques           = None
        self.ques_mask      = None
        self.contx          = None
        self.contx_mask     = None
        self.ans1           = None
        self.ans2           = None
        self.ans1_mask      = None
        self.ans2_mask      = None
        ## These for LSTM/GRU
        self.ans1_len       = None
        self.ans2_len       = None


        self.n_exchange     = 0


    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'ques'          : self.ques[idx],
            'ques_mask'     : self.ques_mask[idx],
            'contx'         : self.contx[idx],
            'contx_mask'    : self.contx_mask[idx],
            'ans1'          : self.ans1[idx],
            'ans2'          : self.ans2[idx],
            'ans1_len'      : self.ans1_len[idx],
            'ans2_len'      : self.ans2_len[idx],
            'ans1_mask'     : self.ans1_mask[idx],
            'ans2_mask'     : self.ans2_mask[idx]
        }

    def bert_service(self, sent: str, max_len: int) -> Tuple[np.array, np.array, int]:
        ## Convert sent in str to tokens
        sent        = sent.split(' ')

        ## convert tokens to id using BertTokenizer
        tok_ids     = bert_tokenizer.convert_tokens_to_ids(sent)

        ## Add CLS and SEP token
        tok_ids     = [cls_id] + tok_ids + [sep_id]
        len_        = len(tok_ids)

        ## Create attention mask
        attn_mask   = [1]*len_ + [0]*(max_len - len_)

        ## Pad 'sent' to max len of paras
        tok_ids     = tok_ids + [pad_id]*(max_len - len_)

        return np.array(tok_ids), np.array(attn_mask), len_

    def f_cvt_tensor(self, entries, queue, arg):
        """ Convert entry to tensors """

        for entry in entries.itertuples():
            ## Process question
            ques, ques_mask, _  = self.bert_service(entry.question, args.seq_len_ques)


            ## Create context and process it
            En          = ast.literal_eval(entry.En)
            Hn          = ast.literal_eval(entry.Hn)

            contx_      = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]
            contx, contx_mask  = [], []
            for para in contx_:
                para, para_mask, _= self.bert_service(para, args.seq_len_para)

                contx.append(para)
                contx_mask.append(para_mask)

            contx       = np.concatenate(contx)
            contx_mask  = np.concatenate(contx_mask)


            ## Process answers
            ans1, ans1_mask, ans1_len   = self.bert_service(entry.answers[0], args.seq_len_ans)
            ans2, ans2_mask, ans2_len   = self.bert_service(entry.answers[1], args.seq_len_ans)


            queue.put({
                'ques'          : ques,
                'ques_mask'     : ques_mask,
                'contx'         : contx,
                'contx_mask'    : contx_mask,
                'ans1'          : ans1,
                'ans2'          : ans2,
                'ans1_len'      : ans1_len,
                'ans2_len'      : ans2_len,
                'ans1_mask'     : ans1_mask,
                'ans2_mask'     : ans2_mask
            })

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        self.ques           = []
        self.ques_mask      = []
        self.contx          = []
        self.contx_mask     = []
        self.ans1           = []
        self.ans2           = []
        self.ans1_len       = []
        self.ans2_len       = []
        self.ans1_mask      = []
        self.ans2_mask      = []

        gc.collect()

        ######################
        # Fill fields
        ######################
        entries = ParallelHelper(self.f_cvt_tensor, df, lambda dat, l, h: dat.iloc[l:h],
                                 args.num_proc).launch()

        for entry in entries:
            self.ques.append(entry['ques'])
            self.ques_mask.append(entry['ques_mask'])
            self.contx.append(entry['contx'])
            self.contx_mask.append(entry['contx_mask'])
            self.ans1.append(entry['ans1'])
            self.ans2.append(entry['ans2'])
            self.ans1_len.append(entry['ans1_len'])
            self.ans2_len.append(entry['ans2_len'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans2_mask.append(entry['ans2_mask'])


    def switch_answerability(self):
        self.n_exchange += 1
