
import torch.nn as torch_nn
from transformers import BertModel

from configs import args

class BertEmbedding(torch_nn.Module):
    """Module to embed paragraphs and question using Bert model."""
    def __init__(self):
        super().__init__()

        self.embedding  = BertModel.from_pretrained(args.bert_model)

    def forward(self, X, X_mask):
        # X, X_mask: [b, *]

        X   = self.embedding(X, X_mask)[1]
        # X: [b, 768]

        return X
