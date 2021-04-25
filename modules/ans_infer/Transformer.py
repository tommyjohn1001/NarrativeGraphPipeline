
import torch.nn as torch_nn
import torch
from torchtext.vocab import Vectors

from configs import args

class TransDecoder(torch_nn.Module):
    def __init__(self, vocab) -> None:
        super().__init__()

        self.SEP_indx   = vocab.stoi(vocab.SEP)
        self.cls_emb    = Vectors("glove.6B.200d.txt", cache=".vector_cache/")\
                            .get_vecs_by_tokens(vocab.CLS)\
                            .unsqueeze(0)\
                            .to(args.device)
        # [1, d_embd]

        self.d_hid1     = args.d_hid * 2
        self.d_embd     = args.d_embd
        self.d_vocab    = args.d_vocab
        self.max_len_ans= args.max_len_ans
        self.seq_len_ans= args.seq_len_ans

        decoder_layer   = torch_nn.TransformerDecoderLayer(d_model=self.d_hid1, nhead=8)
        self.decoder    = torch_nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.ff_ans     = torch_nn.Sequential(
            torch_nn.Linear(self.d_embd, self.d_hid1),
            torch_nn.ReLU(),
            torch_nn.Dropout(args.dropout)
        )

        self.ff_pred    = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid1, self.d_vocab),
            torch_nn.ReLU(),
            torch_nn.Dropout(args.dropout)
        )

    def get_mask_sep(self, pred):
        # X : [b, seq_len_ans, d_vocab]

        batch   = pred.shape[0]


        indx = torch.argmax(pred, dim=2)
        # [b, seq_len_ans]

        SEP_indices = []
        for b in range(batch):
            for i in range(indx.shape[1]):
                if indx[b, i].item() == self.SEP_indx:
                    break
            SEP_indices.append(i)

        mask = []
        for b in range(batch):
            mask.append(torch.Tensor([1]*(SEP_indices[b]) +
                                     [0]*(self.seq_len_ans - SEP_indices[b])).unsqueeze(0))

        mask = torch.vstack(mask).to(args.device)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.d_vocab)

        return mask

    def forward(self, Y, ans, ans_mask, is_inferring=False):
        # Y         : [b, seq_len_contx, d_hid * 2]
        # ans_mask  : [b, seq_len_ans]
        # ans       : [b, seq_len_ans, d_embd]

        batch   = ans_mask.shape[0]

        # Add CLS embedding to ans
        cls_emb = self.cls_emb.unsqueeze(0).repeat(1, batch, 1)
        cls_emb = self.ff_ans(cls_emb)
        # [1, b, d_hid1]

        Y       = Y.transpose(0, 1)
        # [seq_len_contx, b, d_hid1]

        if is_inferring:
            pred    = torch.zeros((batch, self.max_len_ans, self.d_vocab)).to(args.device)

            last_tok= cls_emb
            for i in range(self.max_len_ans):

                output  = self.decoder(last_tok, Y)
                # [1, b, d_hid1]

                last_tok= output

                output  = output.transpose(0, 1)
                # [b, 1, d_hid1]

                output  = self.ff_pred(output)
                # [b, 1, d_vocab]

                pred[:, i, :] = output.squeeze(1)

            pad = torch.zeros((batch, self.seq_len_ans - self.max_len_ans, self.d_vocab)).to(args.device)
            pred= torch.cat((pred, pad), dim=1)
            # [b, seq_len_ans, d_vocab]

        else:            
            ans     = self.ff_ans(ans).transpose(0, 1)
            # [seq_len_ans, b, d_embd]
            ans     = torch.cat((cls_emb, ans), dim=0)
            # [1 + seq_len_ans, b, d_embd]

            pred    = self.decoder(ans, Y)[:-1, :, :]
            # [seq_len_ans, b, d_hid1]
            pred    = pred.transpose(0, 1)
            # [b, seq_len_ans, d_hid1]

            pred    = self.ff_pred(pred)
            # [b, seq_len_ans, d_vocab]


        ########################
        # Multiply 'pred' with 2 masks
        ########################
        # Multiply 'pred' with 'ans_mask' to ignore masked position in tensor 'pred'
        ans_mask    = ans_mask.unsqueeze(-1).repeat(1, 1, self.d_vocab)
        pred        = pred * ans_mask
        # pred: [b, seq_len_ans, d_vocab]

        # Multiply 'pred' with mask SEP
        sep_mask    = self.get_mask_sep(pred)
        pred        = pred * sep_mask
        # pred: [b, seq_len_ans, d_vocab]


        return pred
