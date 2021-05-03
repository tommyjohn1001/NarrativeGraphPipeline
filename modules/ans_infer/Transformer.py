
from transformers import BertModel
import torch.nn as torch_nn
import torch

from modules.narrativepipeline.utils import Vocab
from modules.ans_infer.utils import BeamSearch
from configs import args

class TransDecoder(torch_nn.Module):
    def __init__(self, vocab: Vocab, embedding) -> None:
        super().__init__()

        self.vocab      = vocab

        self.d_hid1     = args.d_hid * 2
        self.d_embd     = args.d_embd
        self.d_vocab    = args.d_vocab
        self.max_len_ans= args.max_len_ans
        self.seq_len_ans= args.seq_len_ans

        decoder_layer   = torch_nn.TransformerDecoderLayer(d_model=self.d_hid1, nhead=8)
        self.decoder    = torch_nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.embedding_ans  = embedding
        self.ff_ans         = torch_nn.Sequential(
            torch_nn.Linear(768, self.d_hid1),
            torch_nn.Tanh(),
            torch_nn.Dropout(args.dropout)
        )

        self.ff_pred    = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid1, self.d_vocab),
            torch_nn.Tanh(),
            torch_nn.Dropout(args.dropout)
        )

    def get_mask_sep(self, pred):
        # X : [b, seq_len_ans, d_vocab]
        SEP_indx= self.vocab.stoi(self.vocab.sep)

        batch   = pred.shape[0]


        indx = torch.argmax(pred, dim=2)
        # [b, seq_len_ans]

        SEP_indices = []
        for b in range(batch):
            for i in range(indx.shape[1]):
                if indx[b, i].item() == SEP_indx:
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
        # ans       : [b, seq_len_ans]

        batch   = ans_mask.shape[0]

        if is_inferring:
            ###############
            # BeamSearch from myself
            ###############
            def infer(tok_ids, Y):
                # Y     : [seq_len_contx, d_hid * 2]
                # tok_ids: list of token ids
                Y      = Y.unsqueeze(0).transpose(0, 1)
                # [seq_len_contx, b=1, d_hid*2]

                toks_emb = torch.FloatTensor(self.vocab.get_vecs_by_tokids(tok_ids))\
                            .unsqueeze(1)\
                            .to(args.device)
                # [seq=*, b=1, d_embd]

                toks_emb = self.ff_ans(toks_emb)
                # [seq=*, b=1, d_hid*2]

                output  = self.decoder(toks_emb, Y)
                # [seq=*, b=1, d_hid*2]

                output  = output.transpose(0, 1)
                # [b=1, seq=*, d_hid*2]

                output  = self.ff_pred(output).squeeze(0)
                # [seq=*, d_vocab]

                return output
            pred        = []
            beam_search = BeamSearch(max_breadth=args.beam_breadth,
                                     model=infer, early_stop=True)

            for b in range(batch):
                indices = beam_search.search(Y[b, :, :])
                print(self.vocab.itos(indices))
                pred_   = torch.zeros((self.seq_len_ans, self.d_vocab))

                for i, indx in enumerate(indices):
                    pred_[i, indx]  = 1

                pred.append(pred_)

            pred    = torch.vstack(pred).to(args.device)
            # [b, seq_len_ans, d_vocab]

        else:
            Y       = Y.transpose(0, 1)
            # [seq_len_contx, b, d_hid1]

            ans     = self.embedding_ans(ans, ans_mask)[0]
            # [b, seq_len_ans, 768]
            ans     = self.ff_ans(ans).transpose(0, 1)
            # [seq_len_ans, b, d_hid1]

            pred    = self.decoder(ans, Y)
            # [seq_len_ans, b, d_hid1]
            pred    = pred.transpose(0, 1)
            # [b, seq_len_ans, d_hid1]

            pred    = self.ff_pred(pred)
            # [b, seq_len_ans, d_vocab]


        ########################
        # Multiply 'pred' with 2 masks
        ########################
        # Multiply 'pred' with 'ans_mask' to ignore masked position in tensor 'pred'
        ans_mask    = ans_mask.unsqueeze(-1).repeat(1, 1, self.d_vocab).to(args.device)
        pred        = pred * ans_mask
        # pred: [b, seq_len_ans, d_vocab]

        # Multiply 'pred' with mask SEP
        sep_mask    = self.get_mask_sep(pred).to(args.device)
        pred        = pred * sep_mask
        # pred: [b, seq_len_ans, d_vocab]


        return pred
