'''This file contain modules for loading data and training procedure. Other component layers
are in other directories.'''
from collections import defaultdict
import glob, ast, gc, json, os

from torch.utils.data import Dataset
from torchtext.vocab import Vectors
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from src.utils import ParallelHelper
from configs import logging, args, PATH


SEQ_LEN_CONTEXT = args.seq_len_para * args.n_paras


class Vocab:
    def __init__(self, path_vocab=None) -> None:
        if path_vocab is None:
            path_vocab  =PATH['vocab']

        self.glove_embd = Vectors("glove.6B.200d.txt", cache=PATH['glove_embd'])

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

    def stoi(self, toks):
        def s_to_id(tok):
            try:
                id_ = self.dict_stoi[tok]
            except KeyError:
                id_ = self.dict_stoi[self.unk]

            return id_

        return list(map(s_to_id, toks))

    def itos(self, ids):
        def id_to_s(id_):
            try:
                tok = self.dict_itos[id_]
            except KeyError:
                tok = self.unk

            return tok

        if isinstance(ids, int):
            return id_to_s(ids)
        if isinstance(ids, list):
            return list(map(id_to_s, ids))
        else:
            raise TypeError(f"'ids' must be 'list' or 'int' type. Got {type(ids)}")

    def conv_ids_to_vecs(self, ids):
        if isinstance(ids, torch.Tensor) or isinstance(ids, np.ndarray):
            ids = ids.tolist()

        def id_to_vec(id_):
            if id_ == self.pad_id:
                return self.pad_vec
            elif id_ == self.cls_id:
                return self.cls_vec
            elif id_ == self.sep_id:
                return self.sep_vec
            else:
                try:
                    return self.glove_embd.get_vecs_by_tokens(self.itos(id_)).numpy()
                except KeyError:
                    return self.unk_vec

        assert isinstance(ids, list), f"ids must be 'list' type. Got {type(ids)}"

        return list(map(id_to_vec, ids))


class CustomDataset(Dataset):
    def __init__(self, path_csv_dir, vocab: Vocab):
        # Search for available data file within directory
        self.file_names = glob.glob(f"{path_csv_dir}/data_*.csv")

        # Read vocab
        self.vocab      = vocab

        self.nlp_bert   = BertTokenizer.from_pretrained(PATH['bert'])

        # self.docId          = None
        # self.ques_plain     = None

        self.ques           = None
        self.ques_mask      = None
        self.ans1           = None
        self.ans2           = None
        self.ans1_mask      = None
        self.ans1_ids       = None
        self.ans2_ids       = None
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
            'ans1_ids'      : self.ans1_ids[idx],
            'ans2_ids'      : self.ans2_ids[idx],
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
        sent_ids_   = self.vocab.stoi(sent_)
        sent_ids_   = [self.vocab.cls_id] + sent_ids_ + [self.vocab.sep_id]

        sent_len_   = len(sent_ids_)
        sent_mask_  = np.array([1]*sent_len_ + [0]*(max_len - sent_len_), dtype=np.float)
        sent_ids_   = sent_ids_ + [self.vocab.pad_id]*(max_len - sent_len_)
        sent_       = self.vocab.conv_ids_to_vecs(sent_ids_)

        return sent_, sent_mask_, np.array(sent_ids_, dtype=np.int)

    def f_process_file_multi(self, entries, queue, arg):
        for entry in entries.itertuples():
            ###########################
            # Process question
            ###########################
            ques, ques_mask, _  = self.process_sent(entry.question, args.seq_len_ques)
            ques    = np.vstack(ques)


            ###########################
            # Process answers
            ###########################
            answers         = ast.literal_eval(entry.answers)

            # This trick ensures training process occurs in longer answer
            if len(' '.split(answers[0])) < len(' '.split(answers[1])):
                answers[0], answers[1] = answers[1], answers[0]

            ans1, ans1_mask, ans1_ids   = self.process_sent(answers[0], args.seq_len_ans)
            ans2, _, ans2_ids           = self.process_sent(answers[0], args.seq_len_ans)
            ans1    = np.vstack(ans1)
            ans2    = np.vstack(ans2)


            ###########################
            # Process context
            ###########################
            En      = ast.literal_eval(entry.En)
            Hn      = ast.literal_eval(entry.Hn)

            contx = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]

            # Process context
            paras, paras_mask = [], []
            for sent in contx:
                sent, sent_mask, _ = self.process_sent(sent, args.seq_len_para)
                paras.append(np.expand_dims(sent, axis=0))
                paras_mask.append(np.expand_dims(sent_mask, axis=0))

            # This piece of code pads zero tensor to 'paras' and 'paras_mask'
            # in case no. paras is less than args.n_paras
            for _ in range(args.n_paras - len(paras)):
                paras.append(np.zeros((1, args.seq_len_para, args.d_embd)))
                paras_mask.append(np.zeros((1, args.seq_len_para)))

            paras       = np.vstack(paras)
            paras_mask  = np.vstack(paras_mask)

            # Pad paras and paras_mask
            if paras.shape[0] < args.n_paras:
                pad = np.zeros((args.n_paras - paras.shape[0], args.seq_len_para))
                paras       = np.concatenate((paras, pad), axis=0) 
                paras_mask  = np.concatenate((paras_mask, pad), axis=0)


            queue.put({
                'ques'          : ques,
                'ques_mask'     : ques_mask,
                'ans1'          : ans1,
                'ans2'          : ans2,
                'ans1_mask'     : ans1_mask,
                'ans1_ids'      : ans1_ids,
                'ans2_ids'      : ans2_ids,
                'paras'         : paras,
                'paras_mask'    : paras_mask
            })

    def f_process_file_single(self, entry):
        ###########################
        # Process question
        ###########################
        ques, ques_mask, _  = self.process_sent(entry.question, args.seq_len_ques)
        ques    = np.vstack(ques)


        ###########################
        # Process answers
        ###########################
        answers         = ast.literal_eval(entry.answers)

        # This trick ensures training process occurs in longer answer
        if len(' '.split(answers[0])) < len(' '.split(answers[1])):
            answers[0], answers[1] = answers[1], answers[0]

        ans1, ans1_mask, ans1_ids   = self.process_sent(answers[0], args.seq_len_ans)
        ans2, _, ans2_ids           = self.process_sent(answers[0], args.seq_len_ans)
        ans1    = np.vstack(ans1)
        ans2    = np.vstack(ans2)


        ###########################
        # Process context
        ###########################
        En      = ast.literal_eval(entry.En)
        Hn      = ast.literal_eval(entry.Hn)

        contx = En[self.n_exchange:args.n_paras] + Hn[:self.n_exchange]

        # Process context
        paras, paras_mask = [], []
        for sent in contx:
            sent, sent_mask, _ = self.process_sent(sent, args.seq_len_para)
            paras.append(np.expand_dims(sent, axis=0))
            paras_mask.append(np.expand_dims(sent_mask, axis=0))

        # This piece of code pads zero tensor to 'paras' and 'paras_mask'
        # in case no. paras is less than args.n_paras
        for _ in range(args.n_paras - len(paras)):
            paras.append(np.zeros((1, args.seq_len_para, args.d_embd)))
            paras_mask.append(np.zeros((1, args.seq_len_para)))

        paras       = np.vstack(paras)
        paras_mask  = np.vstack(paras_mask)


        return {
            'ques'          : ques,
            'ques_mask'     : ques_mask,
            'ans1'          : ans1,
            'ans2'          : ans2,
            'ans1_mask'     : ans1_mask,
            'ans1_ids'      : ans1_ids,
            'ans2_ids'      : ans2_ids,
            'paras'         : paras,
            'paras_mask'    : paras_mask
        }

    def read_shard(self, path_file):
        df  = pd.read_csv(path_file, index_col=None, header=0)

        # self.docId          = []
        # self.ques_plain     = []
        self.ques           = []
        self.ques_mask      = []
        self.ans1           = []
        self.ans2           = []
        self.ans1_mask      = []
        self.ans1_ids       = []
        self.ans2_ids       = []
        self.paras          = []
        self.paras_mask     = []

        gc.collect()

        ######################
        # Fill self.ques, self.ans1,  self.ans2,
        # answers' mask and index
        ######################
        if args.num_proc > 1:
            entries = ParallelHelper(self.f_process_file_multi, df, lambda dat, l, h: dat.iloc[l:h],
                                    args.num_proc).launch()
        else:
            entries = list(map(self.f_process_file_single, tqdm(df.itertuples(), total=len(df), desc=os.path.basename(path_file))))

        for entry in entries:
            # self.docId.append(entry['docId'])
            # self.ques_plain.append(entry['ques_plain'])
            self.ques.append(entry['ques'])
            self.ques_mask.append(entry['ques_mask'])
            self.ans1.append(entry['ans1'])
            self.ans2.append(entry['ans2'])
            self.ans1_mask.append(entry['ans1_mask'])
            self.ans1_ids.append(entry['ans1_ids'])
            self.ans2_ids.append(entry['ans2_ids'])
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
        for word in words:
            vocab_file.write(word + '\n')
