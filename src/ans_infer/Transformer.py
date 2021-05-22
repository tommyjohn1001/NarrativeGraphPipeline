
import torch.nn as torch_nn
import torch

from src.finegrained.FineGrain import FineGrain
from src.narrativepipeline.utils import Vocab
from src.ans_infer.utils import BeamSearch
from configs import args

class TransDecoder(torch_nn.Module):
    def __init__(self, vocab: Vocab, embd_layer: FineGrain) -> None:
        super().__init__()

        self.vocab      = vocab

        self.d_hid      = args.d_hid
        self.d_vocab    = args.d_vocab
        self.max_len_ans= args.max_len_ans
        self.seq_len_ans= args.seq_len_ans

        decoder_layer   = torch_nn.TransformerDecoderLayer(d_model=self.d_hid, nhead=args.trans_nheads)
        self.decoder    = torch_nn.TransformerDecoder(decoder_layer, num_layers=args.trans_nlayers)

        self.embd_layer = embd_layer

        self.ff_pred    = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid, self.d_vocab),
            torch_nn.GELU(),
            torch_nn.Dropout(args.dropout)
        )

    def forward(self, Y, ans, is_inferring=False):
        # Y         : [b, n_nodes + n_paras*seq_len_para, d_hid]
        # ans       : [b, seq_len_ans, d_hid]

        batch   = Y.shape[0]

        if is_inferring:
            def infer(tok_ids, Y):
                # Y      : [n_paras+1, d_hid]
                # tok_ids: list of token ids
                Y      = Y.unsqueeze(1)
                # [n_paras+1, b=1, d_hid]

                # toks_emb = torch.LongTensor(tok_ids).unsqueeze(0).to(args.device)
                # toks_emb = self.embedding(toks_emb)[0]
                # if len(toks_emb.shape) == 2:
                #     toks_emb = toks_emb.unsqueeze(0)
                toks_emb    = self.vocab.conv_ids_to_vecs(list(tok_ids))
                toks_emb    = torch.vstack([torch.from_numpy(embd) for embd in toks_emb]).unsqueeze(0).to(args.device)
                toks_mask   = torch.ones((1, len(tok_ids))).to(args.device)
                # [b=1, seq=*, 200]

                toks_emb = self.embd_layer.linear1(toks_emb.float())
                toks_emb = self.embd_layer.encode_ans(toks_emb, toks_mask).transpose(0, 1)
                # [seq=*, b=1, d_hid]

                output  = self.decoder(toks_emb, Y)
                # [seq=*, b=1, d_hid]

                output  = output.transpose(0, 1)
                # [b=1, seq=*, d_hid]

                output  = self.ff_pred(output).squeeze(0)
                # [seq=*, d_vocab]

                return output
            pred        = []
            beam_search = BeamSearch(beam_size=args.beam_size, max_len=self.max_len_ans,
                                     model=infer, no_repeat_ngram_size=args.n_gram_beam,
                                     topk_strategy="select_nucleus_sample_nobeam")

            for b in range(batch):
                indices = beam_search.search(Y[:, b, :])

                pred.append(indices)

            return pred

        else:
            ans     = ans.transpose(0, 1)
            # [seq_len_ans, b, d_hid]

            Y       = Y.transpose(0, 1)
            # [n_nodes + n_paras*seq_len_para, b, d_hid]

            pred    = self.decoder(ans, Y)
            # [seq_len_ans, b, d_hid]
            pred    = pred.transpose(0, 1)
            # [b, seq_len_ans, d_hid]

            pred    = self.ff_pred(pred)
            # [b, seq_len_ans, d_vocab]


            return pred
