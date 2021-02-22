import glob
import json
import re
import os

from transformers import BertTokenizer
from datasets import load_dataset
import pandas as pd
import spacy

from modules.utils import ParallelHelper, save_object
from configs import args, logging, PATH

BERT_TOKENIZER  = f"{args.init_path}/_pretrained/BERT/{args.bert_model}-vocab.txt"

nlp = spacy.load("en_core_web_sm")

# pandarallel.initialize(nb_workers=args.num_proc, progress_bar=True, verbose=0)


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
            document['question_text']   = document['question']['text']
            document['question_tokens'] = self.BERT_tokenizer.convert_tokens_to_ids(document['question']['tokens'])
            document['answer1_text']    = document['answers'][0]['text']
            document['answer1_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][0]['tokens'])
            document['answer2_text']    = document['answers'][1]['text']
            document['answer2_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][1]['tokens'])


            queue.put(document)


    def f_findGolden2(self, document: dict):
        """Function later used in Pool of multiprocessing. This function export entities
        and find golden passage for each paragraphs. Bert Tokenizer is used also.

        Args:
            document (list): document of data

        Returns:
            dict: dict containing question and golden paras.
        """

        # print(document)

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
        document['question_text']   = document['question']['text']
        document['question_tokens'] = self.BERT_tokenizer.convert_tokens_to_ids(document['question']['tokens'])
        document['answer1_text']    = document['answers'][0]['text']
        document['answer1_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][0]['tokens'])
        document['answer2_text']    = document['answers'][1]['text']
        document['answer2_tokens']  = self.BERT_tokenizer.convert_tokens_to_ids(document['answers'][1]['tokens'])

        del document['document']
        del document['question']
        del document['answers']

        return document


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

            ## NOTE: Method 1
            # # ### Load shard from path
            # dataset = load_dataset('pandas', data_files=path)['train']

            # # ### Process shard of dataset
            # dataset = dataset.map(self.f_findGolden2, num_proc=args.num_proc,
            #                       remove_columns=['document', 'question', 'answers'])


            # save_object(path_storeDat, pd.DataFrame(dataset), is_dataframe=True)


            ## NOTE: Method 2
            # # ### Load shard from path
            # dataset = pd.read_pickle(path)

            # # ### Process shard of dataset

            # added_columns   = ['doc_id', 'doc_tokens', 'question_text',
            #                    'question_tokens', 'question_tokens', 'answer1_text',
            #                    'answer1_tokens', 'answer2_text', 'answer2_tokens']
            # for column in added_columns:
            #     dataset[column] = [None]*len(dataset)

            # ParallelHelper(self.f_findGolden, dataset,
            #                lambda data, lo_bound, hi_bound: data.iloc[lo_bound:hi_bound],
            #                args.num_proc).launch()

            # dataset.drop(axis=1, columns=['document', 'question', 'answers'])


            # save_object(path_storeDat, dataset, is_dataframe=True)


            ## NOTE: Method 3
            # ### Load shard from path
            dataset = load_dataset('pandas', data_files=path)['train']

            # ### Process shard of dataset
            dataset = ParallelHelper(self.f_findGolden, dataset,
                                     lambda data, lo_bound, hi_bound: data.select(range(lo_bound, hi_bound)),
                                     args.num_proc).launch()

            dataset = pd.DataFrame(dataset).drop(axis=1, columns=['document', 'question', 'answers'])
            
            logging.info(f"= Save file: {path_storeDat}")
            save_object(path_storeDat, dataset, is_dataframe=True)
