"""This file contains code reading raw data and do some preprocessing"""
import re, json, os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import unidecode

from src.utils.utils import ParallelHelper
from src.utils import utils

log = utils.get_logger()


class ContextProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        len_para_processing: int = 120,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
        num_workers: int = 4,
    ):
        self.nlp_sacpy = nlp_spacy

        self.path_raw_data = path_raw_data
        self.len_para_processing = len_para_processing
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.num_workers = num_workers

    def read_contx(self, id_):
        with open(
            f"{self.path_raw_data}/texts/{id_}.content", "r", encoding="iso-8859-1"
        ) as d_file:
            return d_file.read()

    def clean_end(self, text: str):
        text = text.split(" ")
        return " ".join(text[:-2])

    def clean_context_movie(self, context: str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", "", context)

        context = re.sub(r" \n ", "\n", context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"(?<=\w|,)\n(?=\w| )", " ", context)
        context = re.sub(r"\n{2,}", "\n", context)

        return context

    def clean_context_gutenberg(self, context: str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", "", context)

        context = re.sub(r"(?<=\w|,|;|-)\n(?=\w| )", " ", context)
        context = re.sub(r"( {2,}|\t)", " ", context)
        # context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"\n{2,}", "\n", context)

        return context

    def export_para(self, toks):
        para_ = []
        for i in range(0, len(toks), self.len_para_processing):
            tmp = re.sub(
                r"( |\t){2,}", " ", " ".join(toks[i : i + self.len_para_processing])
            ).strip()
            para_.append(tmp)

        return np.array(para_)

        # return re.sub(r'( {2,}|\t)', ' ', ' '.join(toks)).strip()

    def extract_html(self, context):
        soup = BeautifulSoup(context, "html.parser")
        context = unidecode.unidecode(soup.text)
        return context

    def extract_start_end(self, context, start, end):
        end = self.clean_end(end)

        start = context.find(start)
        end = context.rfind(end)
        if start == -1:
            start = 0
        if end == -1:
            end = len(context)
        if start >= end:
            start, end = 0, len(context)

        context = context[start:end]

        return context

    def f_process_contx(self, entries, queue):
        for entry in entries.itertuples():
            docId = entry.document_id

            path = self.path_processed_contx.replace("[ID]", docId)
            if os.path.exists(path):
                queue.put(1)
                continue

            context = self.read_contx(docId)
            kind = entry.kind
            start = entry.story_end
            end = entry.story_end

            ## Extract text from HTML
            context = self.extract_html(context)

            ## Use field 'start' and 'end' provided
            context = self.extract_start_end(context, start, end)

            ## Clean context and split into paras
            if kind == "movie":
                sentences = self.clean_context_movie(context).split("\n")
            else:
                sentences = self.clean_context_gutenberg(context).split("\n")

            tokens, paras = np.array([]), np.array([])
            for sent in sentences:
                ## Tokenize, remove stopword
                tokens_ = [
                    tok.text
                    for tok in self.nlp_sacpy(sent)
                    if not (tok.is_stop or tok.is_punct)
                ]
                tokens = np.concatenate((tokens, tokens_))
            len_para = self.len_para_processing
            for i in range(0, len(tokens), len_para):
                para = " ".join(tokens[i : i + len_para])
                para = re.sub(r"( |\t){2,}", " ", para).strip()

                paras = np.concatenate((paras, [para]))

            with open(path, "w+") as contx_file:
                json.dump(paras.tolist(), contx_file, indent=2, ensure_ascii=False)

            queue.put(1)

    def trigger_process_contx(self):
        log.info(" = Process context.")

        documents = pd.read_csv(
            f"{self.path_raw_data}/documents.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            log.info(f" = Process context of split: {split}")

            path_dir = os.path.dirname(self.path_processed_contx)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir, exist_ok=True)

            ParallelHelper(
                self.f_process_contx,
                documents[documents["set"] == split],
                lambda d, l, h: d.iloc[l:h],
                self.num_workers,
            ).launch()


class EntryProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        n_paras: int = 30,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
        num_workers: int = 4,
    ):
        self.nlp_sacpy = nlp_spacy

        self.n_paras = n_paras
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
        tok = [
            tok.text
            for tok in self.nlp_sacpy(text)
            if not (tok.is_stop or tok.is_punct)
        ]

        return " ".join(tok)

    def find_golden(self, query, wm: list) -> set:

        ## Calculate score of query for each para
        query_ = np.expand_dims(wm[query], 0)
        wm_ = wm[:-3]

        scores = cosine_similarity(query_, wm_).squeeze(0)

        goldens = set()

        for ith, score in enumerate(scores):
            if score > 0:
                goldens.add(ith)
                if len(goldens) >= self.n_paras:
                    break

        if len(goldens) < self.n_paras:
            for ith, score in enumerate(scores):
                if score == 0:
                    goldens.add(ith)
                    if len(goldens) >= self.n_paras:
                        break

        return goldens

    def read_processed_contx(self, id_):
        path = self.path_processed_contx.replace("[ID]", id_)
        assert os.path.isfile(path), f"Context with id {id_} not found."
        with open(path, "r") as d_file:
            return json.load(d_file)

    def f_process_entry_single(self, entry):
        """This function is used to run in parallel tailored for list mapping."""

        paras = self.read_processed_contx(entry.document_id)

        #########################
        ## Preprocess question and answer
        #########################
        ques = self.process_ques_ans(entry.question)
        ans1 = self.process_ques_ans(entry.answer1)
        answ2 = self.process_ques_ans(entry.answer2)

        #########################
        ## TfIdf vectorize
        #########################
        tfidfvectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            ngram_range=(1, 3),
            max_features=500000,
        )

        wm = tfidfvectorizer.fit_transform(paras + [ques, ans1, answ2]).toarray()

        question, answer1, answer2 = len(wm) - 3, len(wm) - 2, len(wm) - 1

        #########################
        ## Find golden paragraphs from
        ## question and answers
        #########################
        golden_paras_ques = self.find_golden(question, wm)

        golden_paras_answ1 = self.find_golden(answer1, wm)
        golden_paras_answ2 = self.find_golden(answer2, wm)
        golden_paras_answ = golden_paras_answ1 | golden_paras_answ2

        return {
            "doc_id": entry.document_id,
            "question": entry.question_tokenized.lower(),
            "answers": [
                entry.answer1_tokenized.lower(),
                entry.answer2_tokenized.lower(),
            ],
            "En": [paras[i] for i in golden_paras_answ],
            "Hn": [paras[i] for i in golden_paras_ques],
        }

    def f_process_entry_multi(self, entries, queue):
        """This function is used to run in parallel tailored for ParallelHelper."""

        for entry in entries.itertuples():
            queue.put(self.f_process_entry_single(entry))

    def trigger_process_entries(self):
        """Start processing pairs of question - context - answer"""
        documents = pd.read_csv(
            f"{self.path_raw_data}/qaps.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            documents_ = documents[documents["set"] == split]
            for shard in range(8):
                ### Need to check whether this shard has already been processed
                path = self.path_data.replace("[SPLIT]", split).replace(
                    "[SHARD]", str(shard)
                )
                if os.path.exists(path):
                    continue

                ## Make dir to contain processed files
                os.makedirs(os.path.dirname(path), exist_ok=True)

                ## Start processing (multi/single processing)
                start_ = len(documents_) // 8 * shard
                end_ = start_ + len(documents_) // 8

                if self.num_workers == 1:
                    list_documents = list(
                        map(
                            self.f_process_entry_single,
                            tqdm(
                                documents_.iloc[start_:end_].itertuples(),
                                total=end_ - start_,
                            ),
                        )
                    )
                else:
                    list_documents = ParallelHelper(
                        self.f_process_entry_multi,
                        documents_.iloc[start_:end_],
                        lambda d, l, h: d.iloc[l:h],
                        self.num_workers,
                    ).launch()

                ## Save processed things to Parquet file
                if len(list_documents) > 0:
                    df = pd.DataFrame(list_documents)

                    df.to_parquet(path)


class Preprocess:
    def __init__(
        self,
        num_workers: int = 4,
        len_para_processing: int = 120,
        n_paras: int = 30,
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
            n_paras=n_paras,
            path_raw_data=path_raw_data,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            num_workers=num_workers,
        )

    def preprocess(self):
        self.contx_processor.trigger_process_contx()
        self.entry_processor.trigger_process_entries()
