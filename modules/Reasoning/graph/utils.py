from collections import defaultdict
from itertools import combinations

import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def graph_construction(paras: list, question: str):
    """Construct graph's edges using raw question and paras

    Args:
        paras (list): list of raw para
        question (str): raw question

    Returns:
        numpy.array: array containing graph's edges
    """

    paras.insert(0, question)

    vertex_s, vertex_d  = [], []
    def add_edge(src, dest):
        vertex_s.append(src)
        vertex_d.append(dest)

        vertex_s.append(dest)
        vertex_d.append(src)

    para_vocab  = defaultdict(set)

    #####################
    # Construct para vocab
    #####################
    for ith, para in enumerate(paras):
        for tok in nlp(para):
            if not (tok.is_stop or tok.is_punct):
                para_vocab[tok.text].add(ith)

    #####################
    # Establish edges from para vocab
    #####################
    for list_paras in para_vocab.values():
        for pair in combinations(list_paras, 2):
            add_edge(*pair)

    node_index  = np.array([vertex_s, vertex_d])

    return node_index
