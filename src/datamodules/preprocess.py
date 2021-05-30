import os
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd,\
    numpy as np,\
    spacy

from src.utils.utils import ParallelHelper

N_SAMPLES   = 5000

class Preprocess():
    def __init__(self,
        num_workers: int = 4,
        seq_len: int = 512 * 5,
        path_vocab: str = None,
        path_data: str = None
    ) -> None:


        self.num_workers    = num_workers
        self.seq_len        = seq_len
        self.path_vocab     = path_vocab
        self.path_data      = path_data


        self.nlp_spacy      = spacy.load("en_core_web_sm")
        assert os.path.isfile(path_vocab), f"Path 'vocab' {path_vocab} not existed."
        self.bert_tokenizer = BertTokenizer(vocab_file=path_vocab)


        self.prerpocess()


    def f_process_multi(self, entries, queue):
        for entry in entries:
            text    = entry['article']
            sents   = self.bert_tokenizer(text, padding='max_length', max_length=self.seq_len)

            queue.put({
                'sent_ids'  : np.array(sents['input_ids'], dtype=np.int),
                'sent_masks': np.array(sents['attention_mask'], dtype=np.int)
            })

    def f_process_single(self, entry):
        text    = entry['article']
        sents   = self.bert_tokenizer(text, padding='max_length', max_length=self.seq_len)

        return {
            'sent_ids'  : np.array(sents['input_ids'], dtype=np.int),
            'sent_masks': np.array(sents['attention_mask'], dtype=np.int)
        }

    def prerpocess(self):
        for splt in ["train", "validation", "test"]:
            path_data   = self.path_data.replace("[SPLIT]", splt)
            if os.path.isfile(path_data):
                continue

            dataset = load_dataset("cnn_dailymail", "3.0.0", split=splt).select(range(N_SAMPLES))

            if self.num_workers == 1:
                list_entries    = list(map(self.f_process_single, tqdm(dataset, total=N_SAMPLES)))
            else:
                list_entries    = ParallelHelper(self.f_process_multi, dataset,
                                                 lambda d, l, h: d.select(range(l, h)),
                                                 self.num_workers, splt).launch()

            df  = pd.DataFrame(list_entries)


            df.to_parquet(path_data)
