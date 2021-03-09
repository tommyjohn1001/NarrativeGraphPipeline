'''This file contains code reading raw data and do some preprocessing'''
import re, ast

from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from bs4 import BeautifulSoup
from scipy import spatial
import pandas as pd
import numpy as np
import spacy

from modules.utils import save_object, ParallelHelper
from configs import args, logging, PATH


nlp             = spacy.load("en_core_web_sm",
                             disable=['ner', 'parser', 'tagger'])
nlp.max_length  = 2500000

MAX_LEN_PARA    = 50
MAX_LEVEL       = 3
MAX_NEIGHBORS   = 3


def clean_text(para:str) -> str:
    para = re.sub(r'(\n|\t)', '', para)
    para = re.sub(r'\s{2,}', ' ', para)

    para = para.strip()

    return para.lower()

def clean_end(text:str):
    text = text.split(' ')
    return ' '.join(text[:-2])

def get_score(v1, v2):
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        return -1

    return 1 - spatial.distance.cosine(v1, v2)

def find_golden(query, paragraphs, level:int, neighbors:set):
    # print(f"{level} - {query}")


    ## if level not exceed MAX_LEVEL, keep calling function recursively
    if level == MAX_LEVEL:
        return

    ## Calculate score of query for each para
    scores = [
        (ith, get_score(paragraphs[query], paragraphs[ith]))
        for ith in range(0, len(paragraphs) - 3)
    ]

    ## Sort paras w.r.t relevant score to query
    scores.sort(reverse=True, key= lambda x: x[1])

    ## while not reaching MAX_NEIGHBORS, keep adding to set if not appearing beforehand in set
    found_  = []
    for k in scores:
        if k[0] not in neighbors:
            neighbors.add(k[0])
            found_.append(k[0])

        if len(found_) == MAX_NEIGHBORS:
            break

    for f in found_:
        find_golden(f, paragraphs, level+1, neighbors)

def f_trigger(documents, queue):
    """This function is used to run in parallel to read and initially preprocess
    raw documents.

    Args:
        document (dict): a document of dataset

    Returns:
        dict: a dict containing new fields with value
    """
    for document in documents:
        context = document['document']['text']

        ## Preprocess
        ## Extract text from HTML
        soup    = BeautifulSoup(context, 'html.parser')
        if soup.pre is not None:
            context = ''.join(list(soup.pre.findAll(text=True)))

        ## Clean and lowercase
        context = clean_text(context)


        start_  = document['document']['start'].lower()
        end_    = document['document']['end'].lower()
        end_    = clean_end(end_)

        start_  = context.find(start_)
        end_    = context.rfind(end_)
        if start_ == -1:
            start_ = 0
        if end_ == -1:
            end_ = len(context)

        context = context[start_:end_]


        ## Tokenize
        context = [tok.text for tok in nlp(context)]


        ## Split into paragraphs
        paras               = []
        paras_preprocessed  = []
        for ith in range(0, len(context), MAX_LEN_PARA):
            para    = ' '.join(context[ith:ith+MAX_LEN_PARA])

            paras.append(para)

            ## Tokenize, remove stopword and lemmatization
            tok_context = []
            for tok in nlp(para):
                if not (tok.is_stop or tok.is_punct):
                    tok_context.append(tok.lemma_)
            paras_preprocessed.append(' '.join(tok_context))


        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3),
                                      max_features=500000)

        paras_preprocessed.append(document['question']['text'].lower())
        paras_preprocessed.append(document['answers'][0]['text'].lower())
        paras_preprocessed.append(document['answers'][0]['text'].lower())


        wm = tfidfvectorizer.fit_transform(paras_preprocessed).toarray()

        question, answer1, answer2 = len(wm) - 3, len(wm) - 2, len(wm) - 1

        neighbor_paras_ques = {question}
        find_golden(question, wm, 0, neighbor_paras_ques)
        neighbor_paras_ques.discard(question)

        neighbor_paras_answ1 = {answer1}
        find_golden(question, wm, 0, neighbor_paras_answ1)
        neighbor_paras_answ2 = {answer2}
        find_golden(question, wm, 0, neighbor_paras_answ2)

        neighbor_paras_answ = neighbor_paras_answ1 | neighbor_paras_answ2
        neighbor_paras_answ.discard(answer1)
        neighbor_paras_answ.discard(answer2)


        queue.put({
            'doc_id'    : document['document']['id'],
            'question'  : ' '.join(document['question']['tokens']),
            'answers'   : [' '.join(x['tokens']) for x in document['answers']],
            'En'        : [paras[i] for i in neighbor_paras_answ],
            'Hn'        : [paras[i] for i in neighbor_paras_ques]
        })

def trigger_reading_data():
    """ Start reading and processing data
    """

    for split in ['validation', 'train', 'test']:
        for shard in range(8):
            logging.info(f"= Preprocess dataset: {split} - shard {shard}")

            dataset = load_dataset('narrativeqa', split=split).shard(8, shard)

            list_documents  = ParallelHelper(f_trigger, dataset,
                                             lambda d, lo, hi: d.select(range(lo, hi)),
                                             num_proc=args.num_proc).launch()

            # list_documents = []
            # for document in tqdm(dataset):
            #     list_documents.append(f_trigger(document))

            path    = PATH['dataset_para'].replace("[SPLIT]", split).replace("[SHARD]", str(shard))
            save_object(path, pd.DataFrame(list_documents))



if __name__ == '__main__':
    logging.info("* Reading raw data and decompose into paragraphs")

    trigger_reading_data()
