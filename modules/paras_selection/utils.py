import json
import glob
import re
import os


from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import spacy

from modules.utils import ParallelHelper, save_object
from configs import args, logging, PATH



Tokenizer   = BertTokenizer.from_pretrained(args.bert_model)
nlp         = spacy.load("en_core_web_sm")

CLS = Tokenizer.cls_token_id
SEP = Tokenizer.sep_token_id
PAD = Tokenizer.pad_token_id

MAX_QUESTION_LEN    = 34
MAX_PARA_LEN        = 475


N_SUBSHARDS         = 20

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


def f_conv(document):
    question    = json.loads(document['question_tokens'])
    paragraphs  = json.loads(document['doc_tokens'])
    trgs        = json.loads(document['goldeness'])

    question_   = pad(question, MAX_QUESTION_LEN)

    list_srcs, list_masks, list_trgs = [], [], []

    for paragraph, trg in zip(paragraphs, trgs):
        for ith in range(0, len(paragraph), MAX_PARA_LEN):
            paragraph_  = pad(paragraph[ith:ith+MAX_PARA_LEN], MAX_PARA_LEN)

            pair    = [CLS] + question_[0] + [SEP] + paragraph_[0] + [SEP]
            mask    = [1]   + question_[1] + [1]   + paragraph_[1] + [1]

            list_srcs.append(pair)
            list_masks.append(mask)
            list_trgs.append(trgs)


    document['srcs']    = list_srcs
    document['masks']   = list_masks
    document['trgs']    = list_trgs

    return document


def conv():
    """Convert golden paras file to tensor file for training
    """
    rmv_column_names = ["doc_tokens", "goldeness",
                        "question_text", "question_tokens", "answer1_text",
                        "answer1_tokens", "answer2_text", "answer2_tokens"]

    for split in ['train', 'test', 'valid']:
        paths   = glob.glob(f"./backup/processed_data/{split}/data_*.csv", recursive=True)

        paths.sort()

        for path in paths:
            logging.info(f"= Process file {path}")

            dataset = load_dataset('csv', data_files=path)['train']

            for sha in range(N_SUBSHARDS):
                logging.info(f"== Shard {sha}")

                dataset_    = dataset.shard(N_SUBSHARDS, sha, contiguous=True)

                dataset_    = dataset_.map(f_conv,
                                           remove_columns=rmv_column_names)

                list_documentIDs    = []
                list_masks          = []
                list_srcs           = []
                list_trgs           = []
                for document in tqdm(dataset_):
                    masks   = document['masks']
                    srcs    = document['srcs']
                    trgs    = document['trgs']

                    list_documentIDs.extend([document['doc_id']]*len(masks))
                    list_srcs.extend(srcs)
                    list_masks.extend(masks)
                    list_trgs.extend(trgs)

                path_saveFile   = PATH['data_training'].replace("[SPLIT]", split)\
                                    .replace("[N_SHARD]", sha)

                save_object(path_saveFile,
                    pd.DataFrame({
                        'doc_id': list_documentIDs,
                        'srcs'  : list_srcs,
                        'masks' : list_masks,
                        'trgs'  : list_trgs
                    }), is_dataframe=True)

            break
