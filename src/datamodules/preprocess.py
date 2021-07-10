"""This file contains code reading raw data and do some preprocessing"""
import re, json, os, logging

from rank_bm25 import BM25Okapi
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import unidecode

from src.utils.utils import ParallelHelper


log = logging.getLogger("spacy")
log.setLevel(logging.ERROR)


class ContextProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        len_para_processing: int = 150,
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
            start = entry.story_start.lower()
            end = entry.story_end.lower()

            ## Extract text from HTML
            context = self.extract_html(context)

            ## Clean context and split into paras
            if kind == "movie":
                context = self.clean_context_movie(context)
            else:
                context = self.clean_context_gutenberg(context)

            ## Use field 'start' and 'end' provided
            sentences = self.extract_start_end(context, start, end).split("\n")

            tokens, paras = np.array([]), np.array([])
            for sent in sentences:
                ## Tokenize
                tokens_ = [tok.text for tok in self.nlp_spacy(sent)]
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
                show_bar=True,
            ).launch()


class EntryProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        n_paras: int = 10,
        path_raw_data: str = None,
        path_processed_contx: str = None,
        path_data: str = None,
        num_workers: int = 4,
    ):
        self.nlp_spacy = nlp_spacy

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
        tok = [tok.text for tok in self.nlp_spacy(text) if not tok.is_punct]

        return " ".join(tok)

    def read_processed_contx(self, id_):
        path = self.path_processed_contx.replace("[ID]", id_)
        assert os.path.isfile(path), f"Context with id {id_} not found."
        with open(path, "r") as d_file:
            return json.load(d_file)

    def f_process_entry_multi(self, entries, queue):
        """This function is used to run in parallel tailored for ParallelHelper."""

        for entry in entries.itertuples():
            queue.put(self.f_process_entry(entry))

    def f_process_entry(self, entry):
        """This function is used to run in parallel tailored for list mapping."""

        paras = np.array(self.read_processed_contx(entry.document_id))

        #########################
        ## Preprocess question and answer
        #########################
        ques = self.process_ques_ans(entry.question)
        ans1 = self.process_ques_ans(entry.answer1)
        ans2 = self.process_ques_ans(entry.answer2)
        ans = ans1 + " " + ans2

        #########################
        ## Initialize BM25
        #########################
        tokenized_corpus = [para.split(" ") for para in paras]
        bm25 = BM25Okapi(tokenized_corpus)

        scores_ques = bm25.get_scores(ques).argsort()[::-1][: self.n_paras]
        scores_ans = bm25.get_scores(ans).argsort()[::-1][: self.n_paras]

        #########################
        ## Find golden paragraphs from
        ## question and answers
        #########################
        return {
            "doc_id": entry.document_id,
            "question": entry.question_tokenized.lower(),
            "answers": [ans1.lower(), ans2.lower()],
            "En": paras[scores_ques],
            "Hn": paras[scores_ans],
        }

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
                end_ = start_ + len(documents_) // 8 if shard < 7 else len(documents_)

                if self.num_workers == 1:
                    list_documents = list(
                        map(
                            self.f_process_entry,
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
                        desc=f"Split {split} - shard {shard}",
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
        len_para_processing: int = 150,
        n_paras: int = 10,
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
        # self.contx_processor.trigger_process_contx()
        self.entry_processor.trigger_process_entries()


if __name__ == "__main__":
    path_utils = "/Users/hoangle/Projects/VinAI/Narrative/Narrative_utils"

    Preprocess(
        num_workers=8,
        len_para_processing=150,
        n_paras=10,
        path_raw_data=f"{path_utils}/raw",
        path_processed_contx=f"{path_utils}/processed/proc_contx/[ID].json",
        path_data=f"{path_utils}/processed/[SPLIT]/data_[SHARD].parquet",
    ).preprocess()
