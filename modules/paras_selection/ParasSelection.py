import os


from transformers import BertModel, BertConfig

import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from configs import args, logging, PATH


BERT_PATH       = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"



class ParasScoring(torch_nn.Module):
    """ Score paras
    """

    def __init__(self):
        super().__init__()

        hidden_size     = BertConfig().hidden_size


        self.embedding  = BertModel.from_pretrained(args)
        self.linear     = torch_nn.Linear(hidden_size, hidden_size // 2)
        self.linear2    = torch_nn.Linear(hidden_size // 2, 1)

    def forward(self, X: torch.Tensor) -> torch.tensor:
        ## X: [batch, seq_len]

        X   = self.embedding(X)
        ## X: [batch, seq_len, hidden_size]

        X   = torch_f.dropout(self.linear(X), 0.2)
        ## X: [batch, seq_len, hidden_size//2]
        


class ParasSelection:
    def __init__(self) -> None:
        pass

    def trigger_train(self):
        pass

    def trigger_inference(self):
        pass

    def save_model(self, model: torch_nn.Module):
        """Save trained model

        Args:
            model (torch_nn.Module): model to be saved
        """
        
        #######################
        ### Try to create folder
        ### if not existed
        #######################
        try:
            os.makedirs(os.path.dirname(PATH['savemodel_ParasSelection']))
        except FileExistsError:
            pass

        #######################
        ### Try to create folder
        ### if not existed
        #######################
