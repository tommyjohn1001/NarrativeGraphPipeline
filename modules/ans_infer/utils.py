
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

class CustomPriorityQueue():
    def __init__(self, max_size, token, top="min") -> None:
        self.queue      = []
        self.queue_new  = []
        self.max_size   = max_size
        self.top        = top

        # Initialize
        for _ in range(max_size):
            self.queue.append((0, (token, )))

    def compare(self, a1, a2):
        if self.top == "max":
            return a1 < a2
        return a1 > a2

    def pop(self):
        if len(self.queue) == 0:
            self.queue.extend(self.queue_new[:self.max_size])
            self.queue_new.clear()

        return self.queue.pop(0)

    def push(self, item):
        if len(self.queue_new) == 0:
            self.queue_new.append(item)
        else:
            for i in range(0, min(self.max_size, len(self.queue_new))):
                if self.compare(self.queue_new[i][0], item[0]):
                    self.queue_new.insert(i, item)
                    break

    def initialize(self, item):
        self.queue.append(item)


    def get_max_item(self):
        return self.pop()

class BeamSearch():
    """Implementation of Beam search for inferring process
    """
    def __init__(self,
        beam_size : int = 10,
        max_len : int = 12,
        model = None,
        init_tok: int = 1,
        stop_tok: int = 2,
        no_repeat_ngram_size: int = 5,
        early_stop: bool = False,
        topk_strategy: str = "topk",
        threshold = 0.65):

        self.max_len        = max_len
        self.model          = model
        self.max_len        = max_len
        self.init_tok       = init_tok
        self.stop_tok       = stop_tok
        self.early_stop     = early_stop
        self.ngram_nonrepeat= no_repeat_ngram_size
        self.topk_strategy  = topk_strategy
        self.threshold      = threshold

        if self.topk_strategy == "select_nucleus_sample_nobeam":
            self.beam_size      = 1
        else:
            self.beam_size      = beam_size


    def search(self, Y: torch.Tensor):
        """Start Beam search given tensor X which is result from previous model

        Args:
            Y (torch.Tensor): Result tensfor from previous model
        """
        # Y: [seq_len_contx, d_hid * 2]

        queue  = CustomPriorityQueue(self.beam_size, self.init_tok, "min")

        for length in range(self.max_len):
            # print(f"Length: {length}")
            for beamth in range(self.beam_size):
                # print(f"Beamth: {beamth}")

                #################################
                # Pop from queue and put into model
                #################################
                accum_prob, beam = queue.pop()

                # Pass beam through model
                # Only do this if last token of beam is not self.stop_tok
                if beam[-1] == self.stop_tok:
                    continue
                output  = self.model(beam, Y)[-1, :]
                # [d_vocab]


                #################################
                # Apply constraints: no word occurs twice within n-gram,
                # no n-gram occurs twice
                #################################
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


                #################################
                # Calculate log_softmax and topk
                #################################
                if self.topk_strategy == "topk":
                    distribution        = torch.log_softmax(output, dim=0)
                    topk_dist, topk_tok = self.select_topk(distribution)
                elif self.topk_strategy == "select_nucleus_sample":
                    distribution        = torch.softmax(output, dim=0)
                    topk_dist, topk_tok = self.select_nucleus_sample(distribution)
                elif self.topk_strategy == "select_nucleus_sample_nobeam":
                    distribution        = torch.softmax(output, dim=0)
                    topk_dist, topk_tok = self.select_nucleus_sample_nobeam(distribution)
                elif self.topk_strategy == "select_mix_beam":
                    distribution        = torch.softmax(output, dim=0)
                    topk_dist, topk_tok = self.select_mix_beam(distribution)
                else:
                    raise TypeError
                # topk_dist, topk_tok: [beam_size]

                # for each dis and token in top-k, create new beam
                for dist_, tok_ in zip(topk_dist, topk_tok):
                    accum_dist_ = accum_prob + dist_.item()
                    new_beam    = beam  + (tok_.item(),)

                    queue.push((accum_dist_, new_beam))

        return queue.get_max_item()[1]

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

    def select_topk(self, distribution):
        topBeam_dist, topBeam_tok = torch.topk(distribution, self.beam_size, 0)
        # [beam_size]

        return topBeam_dist, topBeam_tok

    def select_nucleus_sample(self, distribution):
        d_vocab = distribution.shape[0]

        ##########################
        # Select topP using Nucleus sampling
        ##########################
        sorted_val, indices = torch.sort(distribution, dim=0, descending=True)

        accum = 0
        for i, val in enumerate(sorted_val):
            if accum <= self.threshold <= accum + val:
                break
            accum += val

        topP_tok    = indices[:i + 1]
        topP_dist   = torch.index_select(distribution, dim=0, index=topP_tok) / accum


        ##########################
        # Randomly select topK
        ##########################
        K       = self.beam_size * 3
        topK    = []

        for _ in range(K):
            r = torch.rand((1,)).item()
            accum = 0
            for tok, culmulative in zip(topP_tok, topP_dist):
                if accum <= r <= accum + culmulative:
                    break
                accum += culmulative
            topK.append(tok)


        topK_dist   = [distribution[i]/accum if i in topK else 0 for i in range(d_vocab)]
        topK_dist   = torch.log_softmax(torch.FloatTensor(topK_dist), dim=0)


        ##########################
        # Select beam_size element from topK
        ##########################
        topBeam_dist, topBeam_tok = torch.topk(topK_dist, self.beam_size, 0)


        return topBeam_dist, topBeam_tok

    def select_nucleus_sample_nobeam(self, distribution):

        ##########################
        # Select topP using Nucleus sampling
        ##########################
        sorted_val, indices = torch.sort(distribution, dim=0, descending=True)

        accum = 0
        for i, val in enumerate(sorted_val):
            if accum <= self.threshold <= accum + val:
                break
            accum += val

        topP_tok    = indices[:i + 1]
        topP_dist   = torch.index_select(distribution, dim=0, index=topP_tok) / accum


        ##########################
        # Randomly select topK
        ##########################



        r = torch.rand((1,)).item()
        accum = 0
        for tok, culmulative in zip(topP_tok, topP_dist):
            if accum <= r <= accum + culmulative:
                break
            accum += culmulative
        topBeam_tok     = [tok]
        topBeam_dist    = torch.log(torch.FloatTensor((culmulative, )))

        return topBeam_dist, topBeam_tok

    def select_mix_beam(self, distribution):
        temperature     = 1.1

        top_dist, top_tok = torch.topk(distribution, 10000, 0)

        top_dist = top_dist / top_dist.sum(dim=0) / temperature

        topBeam_dist, topBeam_tok = torch.topk(top_dist, self.beam_size, 0)
        topBeam_tok = top_tok[topBeam_tok]
        topBeam_dist= torch.log_softmax(topBeam_dist, dim=0)

        # indx = torch.randint(0, 1000, (1,))
        # topBeam_dist = torch.log(top_dist[indx])
        # topBeam_tok  = top_tok[indx]

        return topBeam_dist, topBeam_tok
