
from collections import defaultdict
import re

from transformers import BertTokenizer
import pandas as pd
import spacy

from modules.utils import load_object, ParallelHelper
from configs import args

BERT_TOKENIZER  = f"{args.init_path}/_pretrained/BERT/{args.bert_model}-vocab.txt"

nlp = spacy.load("en_core_web_sm")


class GoldenParas():
    def __init__(self) -> None:
        
        ## Read file containing raw paragraphs and file containing
        ## question and answer
        path_rawParas       = f"{args.init_path}/_data/QA/NarrativeQA/backup_folder/raw_paragraphs.pkl"
        self.dict_rawParas  = load_object(path_rawParas)

        path_summaries      = f"{args.init_path}/_data/QA/NarrativeQA/summaries.xlsx"
        self.summaries      = pd.read_excel(path_summaries, "qaps", engine="openpyxl")
        self.summaries      = self.summaries.values.tolist()

        ## Load BERT tokenizer
        self.BERT_tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    def extract_entities(self, question: str, answer1: str, answer2: str) -> set:
        """Extract entities from question and answers.

        Args:
            question (str): question
            answer1 (str): First answer
            answer2 (str): Second answer

        Returns:
            set: set containing entities.
        """

        entities    = set()

        def add_(s: set, word: object):
            if word.lemma_ != '-PRON-':
                s.add(word.lemma_)

        def clean_nounPhrase(phrase:str) -> set:
            """ Cleaning by doing the following:
            - remove stopword
            - extract supporting words
            """
            tokens  = set()
            for token in nlp(phrase):
                if  token.pos_ in ['VERB', 'NUM', 'NOUN', 'ADJ'] and\
                    token.lemma_ != '-PRON-':
                    tokens.add(token.lemma_)

            return tokens

        for ob in [answer1, answer2, question]:
            doc = nlp(str(ob))

            for chunk in doc.noun_chunks:
                add_(entities, chunk.root)
                entities = entities.union(clean_nounPhrase(chunk.text))

            for ent in doc.ents:
                add_(entities, ent)

            for token in doc:
                if token.pos_ in ['VERB', 'NUM', 'NOUN', 'ADJ']:
                    add_(entities, token)

        return entities

    # def f_kernel(data: list, queue):
    #     for data_point in data:
    #         queue.put(PreprocessingHelper(data_point).preprocessed_datapoint)
    # def f_findGolden(self, row: list) -> dict:
    def f_findGolden(self, data: list, queue):
        """Function later used in Pool of multiprocessing. This function export entities
        and find golden passage for each paragraphs. Bert Tokenizer is used also.

        Args:
            row (list): row of data

        Returns:
            dict: dict containing question and golden paras.
        """
        for row in data:
            id_doc, question, answer1, answer2  = row[0], row[2], row[3], row[4]

            ##################
            ### Get entities from question
            ### and answers
            ##################
            entities            = self.extract_entities(question, answer1, answer2)

            ##################
            ### Start finding golden passages
            ##################
            golden_paragraphs   = list()
            for paragraph in self.dict_rawParas[id_doc]:
                n_found_entities    = 0

                for ent in entities:
                    try:
                        if re.search(ent, paragraph):
                        # if paragraph.find(ent):
                            n_found_entities    += 1
                    except re.error:
                        pass

                if n_found_entities > 1:
                    para_tokenized  = self.BERT_tokenizer.tokenize(paragraph)
                    golden_paragraphs.append(self.BERT_tokenizer.convert_tokens_to_ids(para_tokenized))



            ##################
            ### Return a dict
            ##################
            queue.put({
                'id_document'   : id_doc,
                "question"      : self.BERT_tokenizer.convert_tokens_to_ids(question.split(' ')),
                "question_plain": question,     ## This field's existence is for keep tracking in later step
                "golden_paras"  : golden_paragraphs
            })

    def generate_goldenParas(self) -> list:
        """Read paragraphs, use keyword method to determine
        golden passages for each question. Additionally, BertTokenizer
        is applied.

        Returns:
            list: list containing questions and corresponding golden paragraphs
        """
        ## For each question - answer - paragraphs triple,
        ## use keyword technique to determine golden passage

        list_golden_paras   = ParallelHelper(self.f_findGolden, self.summaries, 4).launch()
        
        return list_golden_paras
