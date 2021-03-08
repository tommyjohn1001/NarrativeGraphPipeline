'''This file contains code reading raw data and do some preprocessing'''
import re

from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
import spacy

from modules.utils import save_object
from configs import args, logging, PATH


nlp         = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])
nlp.max_length = 2500000

MAX_LEN_PARA    = 50


def clean_text(para:str) -> str:
    para = re.sub(r'(\n|\t)', '', para)
    para = re.sub(r'\s{2,}', ' ', para)

    para = para.strip()

    return para.lower()


def f_trigger(document):
    """This function is used to run in parallel to read and initially preprocess
    raw documents.

    Args:
        document (dict): a document of dataset

    Returns:
        dict: a dict containing new fields with value
    """
    context = document['document']['text']

    ## Preprocess

    ## Extract text from HTML
    soup    = BeautifulSoup(context, 'html.parser')
    if soup.pre is not None:
        context = ''.join(list(soup.pre.findAll(text=True)))

    ## Clean and lowercase
    context = clean_text(context)

    start_  = document['document']['start']
    end_    = document['document']['start']
    start_  = context.find(start_)
    end_    = context.rfind(end_)

    if start_ == -1:
        start_ = 0
    if end_ == -1:
        end_ = len(context)
    context = context[start_:end_]


    ## Tokenize
    tok_context = [tok.text for tok in nlp(context)]
    ## NOTE: The followings are to lemmatize and remove punctuation. Only used in cosine similarity
    ## Tokenize, remove stopword and lemmatization
    # tok_context = []
    # for tok in context:
    #     if not (tok.is_stop or tok.is_punct):
    #         tok_context.append(tok.lemma_)

    ## Split into paragraphs
    paragraphs  = []
    for ith in range(0, len(tok_context), MAX_LEN_PARA):
        paragraphs.append(tok_context[ith:ith+MAX_LEN_PARA])


    document['doc_id']      = document['document']['id']
    document['paragraphs']  = paragraphs
    document['question']    = document['question']['tokens']
    document['answers']     = [x['tokens'] for x in document['answers']]


    return document


def trigger_reading_data():
    """ Start reading and processing data
    """

    for split in ['validation', 'train', 'test']:
        for shard in range(8):
            logging.info(f"= Preprocess dataset: {split} - shard {shard}")

            dataset = load_dataset('narrativeqa', split=split).shard(8, shard)

            dataset = dataset.map(f_trigger, num_proc=args.num_proc,
                                remove_columns=dataset.column_names)

            path    = PATH['dataset_para'].replace("[SPLIT]", split).replace("[SHARD]", str(shard))

            save_object(path, pd.DataFrame(dataset))



if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    trigger_reading_data()
