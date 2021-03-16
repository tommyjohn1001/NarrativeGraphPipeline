import torch.nn.functional as F
import torch.nn as torch_nn
import torch

class NarrativePipeline():
    def __init__(self):
        super().__init__()

    def train(self, model, iterator_train, loss, optimizer):
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
    narrative_pipeline  = NarrativePipeline()

    narrative_pipeline.trigger_train()

    narrative_pipeline.trigger_infer()