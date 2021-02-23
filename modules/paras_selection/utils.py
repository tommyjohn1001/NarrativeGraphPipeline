import glob
import re
import os

from transformers import BertTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import spacy
from transformers.utils.dummy_pt_objects import LineByLineWithSOPTextDataset

from modules.utils import ParallelHelper, save_object
from configs import args, logging, PATH



Tokenizer   = BertTokenizer.from_pretrained(args.bert_model)
nlp         = spacy.load("en_core_web_sm")

CLS = Tokenizer.cls_token_id
SEP = Tokenizer.sep_token_id
PAD = Tokenizer.pad_token_id

MAX_QUESTION_LEN    = 34
MAX_PARA_LEN        = 475

class GoldenParas():
    def __init__(self) -> None:

        ## Read file containing raw paragraphs
        self.paths_rawParas = glob.glob("./backup/dataset_para/dataset_para_*")
        self.paths_rawParas.sort()

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


    def f_findGolden(self, data: pd.DataFrame, queue):
        """Function later used in Pool of multiprocessing. This function export entities
        and find golden passage for each paragraphs. Bert Tokenizer is used also.

        Args:
            document (list): document of data

        Returns:
            dict: dict containing question and golden paras.
        """
        for document in data:
            paragraphs  = document['document']['text']
            question    = document['question']['text']
            answer1     = document['answers'][0]['text']
            answer2     = document['answers'][1]['text']

            ##################
            ### Get entities from question
            ### and answers
            ##################
            entities            = self.extract_entities(question, answer1, answer2)


            ##################
            ### Start finding golden passages
            ##################

            ### This list contains tokenized and converted-to-ids paragraphs
            tokenized_paragraphs    = list()
            ### List contains goldeness of paragraphs
            goldeness               = list()

            for paragraph in paragraphs:
                #### Determine goldeness of paragraph
                n_found_entities    = 0

                for ent in entities:
                    try:
                        if re.search(ent, paragraph):
                        # if paragraph.find(ent):
                            n_found_entities    += 1
                    except re.error:
                        pass

                if n_found_entities > 1:
                    goldeness.append(1)
                else:
                    goldeness.append(0)

                #### Tokenize and convert to id foreach paragraph
                para_tokenized  = self.BERT_tokenizer.tokenize(paragraph)
                tokenized_paragraphs.append(self.BERT_tokenizer.convert_tokens_to_ids(para_tokenized))


            ##################
            ### Update fields in 'document'
            ##################
            document['doc_id']          = document['document']['id']
            document['doc_tokens']      = tokenized_paragraphs
            document['goldeness']       = goldeness
            document['question_text']   = document['question']['text']
            document['question_tokens'] = self.BERT_tokenizer.convert_tokens_to_ids(document['question']['tokens'])
            document['answer1_text']    = document['answers'][0]['text']
            document['answer1_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][0]['tokens'])
            document['answer2_text']    = document['answers'][1]['text']
            document['answer2_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][1]['tokens'])


            queue.put(document)


    def generate_goldenParas(self) -> list:
        """Read paragraphs, use keyword method to determine
        golden passages for each question. Additionally, BertTokenizer
        is applied.

        Args:
            i (int): the order of file to start processing

        Returns:
            list: list containing questions and corresponding golden paragraphs
        """
        ## For each document, use keyword technique
        ## to determine golden passage
        for path in self.paths_rawParas:
            split, n_shard  = re.findall(r"\_(train|test|valid)\_(\d+)\.pkl", path)[0] 
            path_storeDat   = PATH['processed_data'].replace("[N_SHARD]", n_shard).replace("[SPLIT]", split)

            ### Check whether this file is processed (folder specified by 'path' existed)
            if os.path.isfile(path_storeDat):
                continue


            logging.info(f"= Process file: {os.path.split(path)[1]}")

            # ### Load shard from path
            dataset = load_dataset('pandas', data_files=path)['train']

            # ### Process shard of dataset
            dataset = ParallelHelper(self.f_findGolden, dataset,
                                     lambda data, lo_bound, hi_bound: data.select(range(lo_bound, hi_bound)),
                                     args.num_proc).launch()

            dataset = pd.DataFrame(dataset).drop(axis=1, columns=['document', 'question', 'answers'])
            
            logging.info(f"= Save file: {path_storeDat}")
            save_object(path_storeDat, dataset, is_dataframe=True)


def pad(tokens: list, pad_len: int) -> tuple:
    """Pad list of tokens

    Args:
        tokens (list): list of tokens to be added PADDING character
        pad_len (int): no. PADDING characters to be added

    Returns:
        tuple: tuple of 2 lists, one is padded list of tokens, another is list of attention mask
    """
    return tokens + [PAD]*(pad_len-len(tokens)), [1]*len(tokens) + [0]*(pad_len-len(tokens))


def create_tensors(document: dict) -> list:
    """Create tensor to train/infer from question and paragraph tokens and goldeness list of each
    document in dataset by truncating and padding. Since paragraph length may exceed
    max length of Bert, truncation is needed. Therefore, return is list.

    Args:
        document (dict): each document in dataset

    Returns:
        list: list of pairs
    """
    question    = document['question_tokens']
    paragraphs  = document['doc_tokens']
    trgs        = document['goldeness']

    question_   = pad(question, MAX_QUESTION_LEN)

    list_src, list_mask, list_trg   = list(), list(), list()
    for paragraph, trg in zip(paragraphs, trgs):
        for ith in range(0, len(paragraph), MAX_PARA_LEN):
            paragraph_  = pad(paragraph[ith:ith+MAX_PARA_LEN], MAX_PARA_LEN)

            pair    = [CLS] + question_[0] + [SEP] + paragraph_[0] + [SEP]
            mask    = [1]   + question_[1] + [1]   + paragraph_[1] + [1]

            list_src.append(np.asarray(pair))
            list_mask.append(np.asarray(mask))
            list_trg.append(np.asarray(trg))


    document['src']         = np.vstack(list_src)
    document['attn_mask']   = np.vstack(list_mask)
    document['trg']         = np.vstack(list_trg)


    return document

## NOTE: Under development
def get_data_for_training():
    for split in ['train', 'test', 'valid']:
        paths   = glob.glob(f"./backup/processed_data/{split}/data_*.pkl", recursive=True)

        if len(paths) > 0:
            paths.sort()
            for shard, path in enumerate(paths[:2]):
                logging.info(f"= Process dataset: {path}")

                dataset = load_dataset('pandas', data_files=path)['train']

                dataset = dataset.map(create_tensors, num_proc=args.num_proc, remove_columns=dataset.column_names)

                path_trainingData   = PATH['data_training'].replace('[SPLIT]', split).replace('[N_SHARD]', str(shard))

                logging.info(f"= Save dataset: {path}")
                save_object(path_trainingData, pd.DataFrame(dataset), is_dataframe=True)
