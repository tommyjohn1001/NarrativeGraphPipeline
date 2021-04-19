
import torch.nn.functional as torch_f
import torch.nn as torch_nn

from configs import args

class EmbeddingLayer(torch_nn.Module):
    def __init__(self, d_embd=args.d_embd, d_hid=args.d_hid):
        super().__init__()

        self.biLSTM_emb     = torch_nn.LSTM(d_embd, d_hid//2, num_layers=args.n_layers,
                                           batch_first=True, bidirectional=True)

    def forward(self, X, X_len):
        # X     : [b, x, d_embd]
        # X_len : [b,]

        X   = torch_f.relu(X)

        X   = torch_nn.utils.rnn.pack_padded_sequence(X, X_len, batch_first=True,
                                                      enforce_sorted=False)
        X   = self.biLSTM_emb(X)[0]
        X   = torch_nn.utils.rnn.pad_packed_sequence(X, batch_first=True)


        # X: [batch, x, d_hid]
        return X[0]
