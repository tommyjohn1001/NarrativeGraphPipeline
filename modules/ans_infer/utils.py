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
        max_breadth : int = 10,
        model = None,
        max_len : int = 12,
        init_tok: int = 0,
        no_repeat_ngram_size: int = 5):
        self.max_depth      = max_depth
        self.max_breadth    = max_breadth
        self.model          = model
        self.max_len        = max_len
        self.init_tok       = init_tok
        self.ngram_nonrepeat= no_repeat_ngram_size

    def search(self, Y: torch.Tensor):
        """Start Beam search given tensor X which is result from previous model

        Args:
            X (torch.Tensor): Result tensfor from previous model
        """
        queue       = []
        i           = 0
        tok         = self.init_tok
        y           = Y
        prob        = 0
        pred        = [tok]
        while i < self.max_len:
            max_depth   = min([self.max_len - i, self.max_depth])
            self.backtrack(max_depth, queue, tok, y, 0, prob, pred)

            max_prob, max_branch = -10e10, None
            for prob, branch in queue:
                if prob > max_prob:
                    max_prob, max_branch = prob, branch

            pred    = max_branch

            i       += max_depth
            queue.clear()
            tok     = pred[-1]
            prob    = max_prob

        return pred[1:] # Remove CLS token


    def backtrack(self, max_depth, queue: list, tok, y, depth, accum_dist: float, accum_tok: list):
        """Backtrack to every nodes of beam search tree.

        Args:
            max_depth (int): [description]
            queue (list): [description]
            tok ([type]): [description]
            y ([type]): [description]
            depth ([type]): [description]
            accum_dist (float): [description]
            accum_tok (list): [description]
        """
        if depth == max_depth:
            queue.append((accum_dist, accum_tok))
        else:
            dist    = self.model(y, tok)
            # dist  : [d_vocab]
            # y     : [n_layers * 2, b=1, d_hid]

            ## Set current word not match with any previous ngram - 1 words
            for i in range(1, min([self.ngram_nonrepeat, len(accum_tok)]) + 1):
                dist[accum_tok[-i]] = -10e10

            ## Set n-gram not match
            if len(accum_tok) > self.ngram_nonrepeat:
                for tok_ in self.KMPSearch(accum_tok[-(self.ngram_nonrepeat - 1):], accum_tok):
                    dist[tok_] = -10e10

            topk_dist, topk_tok = torch.topk(dist, self.max_breadth, 0)
            # topk_dist, topk_tok: [max_breadth]


            for dist_, tok_ in zip(topk_dist, topk_tok):
                accum_dist_ = accum_dist + dist_.item()
                accum_tok_  = accum_tok  + [tok_.item()]
                self.backtrack(max_depth, queue, tok_, y, depth + 1, accum_dist_, accum_tok_)

    def KMPSearch(self, pat, txt):
        def computeLPSArray(pat, M, lps):
            len = 0 # length of the previous longest prefix suffix
        
            lps[0] # lps[0] is always 0
            i = 1
        
            # the loop calculates lps[i] for i = 1 to M-1
            while i < M:
                if pat[i]== pat[len]:
                    len += 1
                    lps[i] = len
                    i += 1
                else:
                    # This is tricky. Consider the example.
                    # AAACAAAA and i = 7. The idea is similar 
                    # to search step.
                    if len != 0:
                        len = lps[len-1]
        
                        # Also, note that we do not increment i here
                    else:
                        lps[i] = 0
                        i += 1

        list_nodes  = []
        M = len(pat)
        N = len(txt)
    
        # create lps[] that will hold the longest prefix suffix 
        # values for pattern
        lps = [0]*M
        j = 0 # index for pat[]
    
        # Preprocess the pattern (calculate lps[] array)
        computeLPSArray(pat, M, lps)
    
        i = 0 # index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1
    
            if j == M:
                # print(f"Found pattern at index {i-j}, right next: {i-j+M} is {txt[i-j+M]}")
                if i-j+M < N:
                    list_nodes.append(txt[i-j+M])
                j = lps[j-1]
    
            # mismatch after j matches
            elif i < N and pat[j] != txt[i]:
                # Do not match lps[0..lps[j-1]] characters,
                # they will match anyway
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1

        return list_nodes
