'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from multiprocessing import Pool
import glob, ast, gc


from torch.utils.data import Dataset
from torchtext.vocab import Vectors
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import torch
from tqdm import tqdm

from configs import args

SEQ_LEN_CONTEXT = args.seq_len_para * args.n_paras

nlp         = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger', "lemmatizer"])
glove_embd  = Vectors("glove.6B.200d.txt", cache=".vector_cache/",
                      url="http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip")


def pad(l, max_len):
    return l + ['']*(max_len - len(l))

def process_context(context:list):
    """ Convert context to matrix """

    # Remove HTML tag
    context = BeautifulSoup(' '.join(context), 'html.parser').get_text()
    # Tokenize context
    context = [tok.text for tok in nlp(context)]
    # Pad context
    context = pad(context, SEQ_LEN_CONTEXT)
    # Embed context by GloVe
    context = glove_embd.get_vecs_by_tokens(context)

    return context

def f_processing(context):
    return process_context(pad(context, SEQ_LEN_CONTEXT))

def f_reset(paras):
    para_En, para_Hn, n_exchange = paras

    context = para_En[n_exchange:] + para_Hn[:n_exchange]
    context = process_context(pad(context, SEQ_LEN_CONTEXT))

    return context


class TrainDataset(Dataset):
    def __init__(self, path_csv_dir):
        file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        dfs = []
        for filename in file_names:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        self.En         = [ast.literal_eval(entry) for entry in df['En']]
        self.Hn         = [ast.literal_eval(entry) for entry in df['Hn']]

        self.questions  = []
        self.answers1   = []
        self.answers2   = []
        self.contexts   = None

        self.n_exchange = 0


        ######################
        # Fill self.questions, self.answers1
        # and self.answers2
        ######################
        for entry in df.itertuples():
            # Process question: tokenize, pad and question mask
            ques        = pad(entry.question.split(' '), args.seq_len_ques)
            ques        = glove_embd.get_vecs_by_tokens(ques)

            # Process answer1 and answer2
            answer1 = pad(entry.answers[0].split(' '), args.seq_len_ans)
            answer2 = pad(entry.answers[1].split(' '), args.seq_len_ans)

            answer1 = glove_embd.get_vecs_by_tokens(answer1)
            answer2 = glove_embd.get_vecs_by_tokens(answer2)

            # Append to self.question, self.answers1 and self.answers2
            self.questions.append(ques)
            self.answers1.append(answer1)
            self.answers2.append(answer2)


        ######################
        # Initialize self.contexts
        ######################
        self.reset_context()


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'question'      : self.questions[idx],
            'context'       : self.contexts[idx],
            'answer1'       : self.answers1[idx],
            'answer2'       : self.answers2[idx]
        }


    def reset_context(self):
        self.contexts = None
        gc.collect()

        with Pool(args.num_proc) as pool:
            self.contexts   = list(tqdm(pool.imap(f_processing, self.En),
                                        desc="Reset train", total=len(self.questions)))
            # self.contexts   = [process_context(pad(context, SEQ_LEN_CONTEXT))
            #                for context in tqdm(self.En, desc="Reset train")]
        self.n_exchange = 1


    def switch_answerability(self):
        self.contexts = []

        # for para_En, para_Hn in tqdm(zip(self.En, self.Hn), desc="Switch Easy-Hard"):
        #     context = para_En[self.n_exchange:] + para_Hn[:self.n_exchange]
        #     context = process_context(pad(context, SEQ_LEN_CONTEXT))

        #     self.contexts.append(context)

        with Pool(args.num_proc) as pool:
            self.contexts   = list(tqdm(pool.imap(f_reset, zip(self.En, self.Hn,
                                                  [self.n_exchange for _ in range(len(self.questions))])),
                                        desc="Switch Easy-Hard", total=len(self.questions)))
        self.n_exchange += 1

        gc.collect()


class EvalDataset(Dataset):
    def __init__(self, path_csv_dir) -> None:
        super().__init__()

        ######################
        # Read datafile
        ######################
        file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        dfs = []
        for filename in file_names:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)


        self.questions  = []
        self.answers1   = []
        self.answers2   = []
        self.contexts   = []


        ######################
        # Fill self.questions, self.answers1
        # self.answers2 and self.contexts
        ######################
        for entry in tqdm(df.itertuples(), desc="Load valid dataset", total=len(df)):
            # Process question: tokenize, pad and question mask
            ques        = pad(entry.question.split(' '), args.seq_len_ques)
            ques        = glove_embd.get_vecs_by_tokens(ques)

            # Process answer1 and answer2
            answer1 = pad(entry.answers[0].split(' '), args.seq_len_ans)
            answer2 = pad(entry.answers[1].split(' '), args.seq_len_ans)

            answer1 = glove_embd.get_vecs_by_tokens(answer1)
            answer2 = glove_embd.get_vecs_by_tokens(answer2)


            context = ast.literal_eval(entry.Hn)
            context = process_context(pad(context, SEQ_LEN_CONTEXT))

            # Append to self.question, self.answers1, self.answers2 and self.contexts
            self.questions.append(ques)
            self.answers1.append(answer1)
            self.answers2.append(answer2)
            self.contexts.append(context)


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'question'      : self.questions[idx],
            'context'       : self.contexts[idx],
            'answer1'       : self.answers1[idx],
            'answer2'       : self.answers2[idx]
        }
