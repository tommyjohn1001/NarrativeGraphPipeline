import torch.nn.functional as F
import torch.nn as torch_nn
import torch


from modules.narrativepipeline.utils_origin import Embeddinglayer
from modules.pg_decoder.utils import build_vocab_PGD
from configs import args, logging

class NarrativePipeline():
    def __init__(self):
        super().__init__()

        self.embd_layer = Embeddinglayer()

        ################################
        # Build vocab for PointerGeneratorDecoder
        ################################
        logging.info("Preparation: Build vocab for PGD")
        build_vocab_PGD()

    def train(self, model, iterator_train, loss, optimizer):
        for batch in range(args.batch):
            # TODO: Use EmbeddingLayer for question, context and answer 1
            # TODO: Use scheduler, clip_grad_norm
            pass

    def test(self, model, iterator_test):
        pass

    def trigger_train(self):
        ###############################
        pass

    def infer(self, model, iterator_infer):
        pass

    def trigger_infer(self):
        pass

if __name__ == '__main__':
    logging.info("* Start NarrativePipeline")


    narrative_pipeline  = NarrativePipeline()

    narrative_pipeline.trigger_train()

    narrative_pipeline.trigger_infer()
