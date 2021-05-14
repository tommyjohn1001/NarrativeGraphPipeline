
import torch.nn as torch_nn
import torch

from modules.narrativepipeline.utils import Vocab
from modules.ans_infer.utils import BeamSearch
from configs import args

class TransDecoder(torch_nn.Module):
    def __init__(self, vocab: Vocab, bert_model) -> None:
        super().__init__()

        self.vocab      = vocab

        self.d_hid      = args.d_hid
        self.d_vocab    = args.d_vocab
        self.max_len_ans= args.max_len_ans
        self.seq_len_ans= args.seq_len_ans

        decoder_layer   = torch_nn.TransformerDecoderLayer(d_model=self.d_hid, nhead=8)
        self.decoder    = torch_nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.embedding  = bert_model
        self.ff_ans     = torch_nn.Linear(768, self.d_hid, bias=False)

        self.ff_pred    = torch_nn.Sequential(
            torch_nn.Linear(self.d_hid, self.d_vocab),
            torch_nn.Tanh(),
            torch_nn.Dropout(args.dropout)
        )

    def get_mask_sep(self, pred):
        # X : [b, seq_len_ans, d_vocab]
        SEP_indx= self.vocab.sep_id

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
        # Y         : [b, seq_len_ques_para, d_hid]
        # ans       : [b, seq_len_ans, d_hid]
        # ans_mask  : [b, seq_len_ans]

        batch   = Y.shape[0]

        if is_inferring:
            def infer(tok_ids, Y):
                # Y      : [seq_len_ques_para, d_hid]
                # tok_ids: list of token ids
                Y      = Y.unsqueeze(0).transpose(0, 1)
                # [seq_len_ques_para, b=1, d_hid]

                toks_emb = torch.LongTensor(tok_ids).unsqueeze(0).to(args.device)
                toks_emb = self.embedding(toks_emb)[0]
                if len(toks_emb.shape) == 2:
                    toks_emb = toks_emb.unsqueeze(0)
                # [b=1, seq=*, 768]

                toks_emb = self.ff_ans(toks_emb).transpose(0, 1)
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
                                     topk_strategy="select_nucleus_sample")

            for b in range(batch):
                indices = beam_search.search(Y[b, :, :])

                pred.append(indices)

            return pred

        else:
            Y       = Y.transpose(0, 1)
            # [seq_len_ques_para, b, d_hid]

            ans     = ans.transpose(0, 1)

            pred    = self.decoder(ans, Y)
            # [seq_len_ans, b, d_hid]
            pred    = pred.transpose(0, 1)
            # [b, seq_len_ans, d_hid]

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
