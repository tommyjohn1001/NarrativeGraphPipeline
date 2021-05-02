import re

import torch.nn as torch_nn
import torch

from modules.utils import transpose


class AttentivePooling(torch_nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = torch_nn.Linear(dim, 1)

    def forward(self, X):
        # X: [batch, y, dim]

        X_      = torch.tanh(X)
        alpha   = torch.softmax(self.linear(X_), dim=1)
        # alpha: [batch, y, 1]
        r       = torch.bmm(transpose(X), alpha)
        # r: [batch, dim, 1]

        return r.squeeze(-1)

class BeamSearch():
    """Implementation of Beam search for inferring process
    """
    def __init__(self,
        max_depth : int = 5,
        max_breadth : int = 5,
        model = None,
        max_len : int = 12,
        init_tok: int = 0,
        stop_tok: int = 2,
        no_repeat_ngram_size: int = 5,
        early_stop: bool = False):
        self.max_depth      = max_depth
        self.max_breadth    = max_breadth
        self.model          = model
        self.max_len        = max_len
        self.init_tok       = init_tok
        self.stop_tok       = stop_tok
        self.early_stop     = early_stop
        self.ngram_nonrepeat= no_repeat_ngram_size


    def search(self, Y: torch.Tensor):
        """Start Beam search given tensor X which is result from previous model

        Args:
            Y (torch.Tensor): Result tensfor from previous model
        """
        # Y: [seq_len_contx, d_hid * 2]

        nth_depth   = 0
        queue       = []
        final_beams = [] # this list contains beams in leaves node of beam search tree

        # Initiate
        queue.append((0, 0, (self.init_tok, )))
        while nth_depth < self.max_len:
            print(nth_depth)
            max_depth   = min([self.max_len - nth_depth, self.max_depth])

            # Do a Beam Search
            while len(queue) != 0:
                depth, accum_prob, beam = queue.pop(0)
                if depth == max_depth:
                    final_beams.append((accum_prob, beam))
                    continue

                # Pass beam through model to
                output  = self.model(beam, Y)[-1, :]
                # [d_vocab]


                disable_words = set()

                # Within n_gram_nonpreeat words, no 2 words are the same
                for n in range(1, min([self.ngram_nonrepeat - 1, len(beam)])):
                    disable_words.add(beam[-n])

                # Form n-1 gram from n - 1 previous words
                if self.ngram_nonrepeat < len(beam):
                    sub_gram = beam[-self.ngram_nonrepeat + 1:]
                else:
                    sub_gram = None

                # Find all next words of sub_gram in beam
                if sub_gram:
                    list_next = self.find_next(sub_gram, beam)

                    disable_words = disable_words | list_next

                # Disable all words in disable list
                for word in disable_words:
                    output[word] = 0

                # Calculate log_softmax and topk
                distribution        = torch.log_softmax(output, dim=0)
                topk_dist, topk_tok = torch.topk(distribution, self.max_breadth, 0)
                # topk_dist, topk_tok: [max_breadth]

                # for each dis and token in top-k, create new beam
                for dist_, tok_ in zip(topk_dist, topk_tok):
                    accum_dist_ = accum_prob + dist_.item()
                    new_beam    = beam  + (tok_.item(),)

                    if self.early_stop and tok_.item() == self.stop_tok:
                        queue.append((max_depth, accum_dist_, new_beam))
                    else:
                        queue.append((depth + 1, accum_dist_, new_beam))

            # Apply greedy method: Find best beam among found ones
            max_pair = max(final_beams, key=lambda pair: pair[0])
            final_beams.clear()
            queue.append((0, 0, max_pair[1]))

            nth_depth   += max_depth

        return max_pair[1][1:]

    def find_next(self, sub_list: list, main_list: list) -> list:
        """Find all occurences of sub_list in main_list and return the number next to the sub_list in main_list.

        Args:
            sub_list (list): list to check
            main_list (list): list to be checked

        Returns:
            list: list of all next numbers of sub_list in main_list
        """
        sub_    = ' '.join(map(str, sub_list))
        main_   = ' '.join(map(str, main_list))

        n_num_sub = sub_.count(' ')

        list_next = []
        for m in re.finditer(sub_, main_):
            idx = m.start()

            n_nums_main = main_[:idx].count(' ')

            next_num = n_num_sub + n_nums_main + 1
            if next_num < len(main_list):
                list_next.append(main_list[next_num])

        return set(list_next)
