"""This file contains code reading raw data and do some preprocessing"""
from collections import defaultdict
from glob import glob
import re, json, os, logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from src.utils.utils import ParallelHelper


log = logging.getLogger("spacy")
log.setLevel(logging.ERROR)


class ContextProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        len_para_processing: int = 500,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
        num_workers: int = 4,
    ):
        self.nlp_spacy = nlp_spacy

        self.path_raw_data = path_raw_data
        self.len_para_processing = len_para_processing
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.num_workers = num_workers

    def f_process_contx(self, entries, queue):
        for entry in entries.itertuples():
            docId = entry.document_id

            path = self.path_processed_contx.replace("[ID]", docId)
            folder_path = os.path.dirname(path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            if os.path.exists(path):
                queue.put(1)
                continue

            paras = []
            tokens = entry.summary_tokenized.split(" ")
            for i in range(0, len(tokens), self.len_para_processing):
                paras.append(" ".join(tokens[i : i + self.len_para_processing]).lower())

            with open(path, "w+") as contx_file:
                json.dump(paras, contx_file, indent=2, ensure_ascii=False)

            queue.put(1)

    def trigger_process_contx(self):
        log.info(" = Process context.")

        documents = pd.read_csv(
            f"{self.path_raw_data}/summaries.csv", header=0, index_col=None
        )

        ParallelHelper(
            self.f_process_contx,
            documents,
            lambda d, l, h: d.iloc[l:h],
            self.num_workers,
            show_bar=True,
        ).launch()


class EntryProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
        num_workers: int = 4,
    ):
        self.nlp_spacy = nlp_spacy

        self.path_raw_data = path_raw_data
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.num_workers = num_workers

    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        toks = [tok.text.lower() for tok in self.nlp_spacy(text) if not tok.is_punct]

        return " ".join(toks), len(toks)

    def read_processed_contx(self, id_):
        path = self.path_processed_contx.replace("[ID]", id_)
        assert os.path.isfile(path), f"Context with id {id_} not found."
        with open(path, "r") as d_file:
            return json.load(d_file)

    def f_process_entry(self, entry):
        """This function is used to run in parallel tailored for list mapping."""
        ques, _ = self.process_ques_ans(entry.question)
        ans1, len_ans1 = self.process_ques_ans(entry.answer1)
        ans2, len_ans2 = self.process_ques_ans(entry.answer2)
        if len_ans1 < len_ans2:
            ans1, ans2 = ans2, ans1

        return {
            "doc_id": entry.document_id,
            "question": ques,
            "answers": [ans1, ans2],
            "context": self.read_processed_contx(entry.document_id),
        }

    def f_process_entry_multi(self, entries, queue):
        """This function is used to run in parallel tailored for ParallelHelper."""

        for entry in entries.itertuples():
            queue.put(self.f_process_entry(entry))

    def trigger_process_entries(self):
        """Start processing pairs of question - context - answer"""
        documents = pd.read_csv(
            f"{self.path_raw_data}/qaps.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            documents_ = documents[documents["set"] == split]

            ### Need to check whether this shard has already been processed
            path = self.path_data.replace("[SPLIT]", split)

            if os.path.exists(path):
                continue

            if self.num_workers == 1:
                list_documents = list(
                    map(
                        self.f_process_entry,
                        tqdm(
                            documents_.itertuples(),
                            total=len(documents_),
                        ),
                    )
                )
            else:

                list_documents = ParallelHelper(
                    self.f_process_entry_multi,
                    documents_,
                    lambda d, l, h: d.iloc[l:h],
                    self.num_workers,
                    desc=f"Split {split}",
                    show_bar=True,
                ).launch()

            ## Save processed things to Parquet file
            if len(list_documents) > 0:
                df = pd.DataFrame(list_documents)

                df.to_parquet(path)


class Preprocess:
    def __init__(
        self,
        num_workers: int = 4,
        len_para_processing: int = 500,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
    ):

        nlp_spacy = spacy.load("en_core_web_sm")
        nlp_spacy.add_pipe("sentencizer")

        ######################
        # Define processors
        ######################
        self.contx_processor = ContextProcessor(
            nlp_spacy=nlp_spacy,
            len_para_processing=len_para_processing,
            path_raw_data=path_raw_data,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            num_workers=num_workers,
        )
        self.entry_processor = EntryProcessor(
            nlp_spacy=nlp_spacy,
            path_raw_data=path_raw_data,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            num_workers=num_workers,
        )

    def preprocess(self):
        self.contx_processor.trigger_process_contx()
        self.entry_processor.trigger_process_entries()


if __name__ == "__main__":
    Preprocess(
        num_workers=6,
        len_para_processing=500,
        path_raw_data="/Users/hoangle/Projects/VinAI/_data/NarrativeQA",
        path_processed_contx="data/proc_contx/[ID].json",
        path_data="data/data_[SPLIT].parquet",
    ).preprocess()
