from collections import defaultdict

import torch.nn as torch_nn
import torch
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy

from modules.data_reading.data_reading import clean_end, clean_text
from configs import logging, args, PATH


class AttentivePooling(torch_nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = torch_nn.Linear(dim, 1)

    def forward(self, X):
        # X: [batch, x, dim]
        batch, seq_len_ques, _ = X.shape

        X_      = torch.tanh(X)
        alpha   = torch.softmax(self.linear(X_), dim=1)
        # alpha: [batch, x, 1]
        r       = torch.bmm(torch.reshape(X, (batch, -1, seq_len_ques)), alpha)
        # r: [batch, dim, 1]

        return r.squeeze(-1)

def build_vocab_PGD():
    """ Build vocab for Pointer Generator Decoder. """
    log = logging.getLogger("spacy")
    log.setLevel(logging.ERROR)

    nlp             = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])
    nlp.max_length  = 2500000

    read_stories    = set()
    word_count      = defaultdict(int)

    answer_toks     = set()

    # Read stories in train, test, valid set
    for split in ["train", "test", "validation"]:
        dataset = load_dataset("narrativeqa", split=split)

        for entry in tqdm(dataset, desc=f"Split '{split}'"):
            if entry['document']['id'] in read_stories:
                continue

            read_stories.add(entry['document']['id'])

            # Process text
            context = entry['document']['text']

            ## Extract text from HTML
            soup    = BeautifulSoup(context, 'html.parser')
            if soup.pre is not None:
                context = ''.join(list(soup.pre.findAll(text=True)))

            ## Clean and lowercase
            context = clean_text(context)


            start_  = entry['document']['start'].lower()
            end_    = entry['document']['end'].lower()
            end_    = clean_end(end_)

            start_  = context.find(start_)
            end_    = context.rfind(end_)
            if start_ == -1:
                start_ = 0
            if end_ == -1:
                end_ = len(context)

            context = context[start_:end_]

            for tok in nlp(context):
                if  not tok.is_punct and\
                    not tok.is_stop and\
                    not tok.like_url:
                    word_count[tok.text] += 1

            # Tokenize answer1 and answer2, answer tokens are added to global vocab
            for tok in entry['answers'][0]['tokens'] + entry['answers'][1]['tokens']:
                answer_toks.add(tok.lower())

    # Filter words
    words = set(w for w, occurence in word_count.items() if occurence >= args.min_count_PGD)

    words = words.union(answer_toks)

    # Write vocab to TXT file
    with open(PATH['vocab_PGD'], 'w+') as vocab_file:
        for word in words:
            vocab_file.write(word + '\n')
