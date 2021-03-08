'''This file contains code reading raw data and do some preprocessing'''
import os, re

from datasets import load_dataset
from bs4 import BeautifulSoup
import spacy

from configs import args, logging, PATH


nlp         = spacy.load("en_core_web_sm")

MAX_LEN_CONTEXT = 2000
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
    ## Preprocess

    ## Trim context
    start_  = document['document']['start']
    start_  = document['document']['text'].find(start_)
    end_    = document['document']['end']
    end_    = document['document']['text'].rfind(end_)

    context = document['document']['text'][start_:end_]

    ## Extract text from HTML
    soup    = BeautifulSoup(context, 'html.parser')
    if soup.pre is not None:
        context = ''.join(list(soup.pre.findAll(text=True)))

    ## Clean and lowercase
    context = clean_text(context)

    ## Tokenize, remove stopword and lemmatization
    context = nlp(context)

    tok_context = [] 

    for tok in context:
        if not (tok.is_stop or tok.is_punct):
            tok_context.append(tok.lemma_)

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
        logging.info("= Preprocess dataset: %s", split)

        dataset = load_dataset('narrativeqa', split=split)

        dataset = dataset.map(f_trigger, num_proc=args.num_proc,
                              remove_columns=dataset.column_names)

        path    = PATH['dataset_para'].replace("[SPLIT]", split)
        if not os.path.isdir(path):
            os.makedirs(path)

        dataset.save_to_disk(path)



if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    trigger_reading_data()
