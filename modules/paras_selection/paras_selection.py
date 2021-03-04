
from modules.paras_selection.ParasSelection import ParasSelection
from modules.paras_selection.utils import GoldenParas, conv
from configs import logging


if __name__ == '__main__':
    logging.info("* Paras Selection training.")

    ########################
    ## Create data for training
    ########################
    ## In this step, we use an heuristic method to filter
    ## golden paragraphs for each question, and use them
    ## to train ParasSelection model then
    logging.info("1. Create golden paras and tensor files for training.")
    GoldenParas().generate_goldenParas()

    logging.info("** Create tensor files for training.")
    conv()

    logging.info("=> Data files are created and saved sufficiently.")


    paras_selection = ParasSelection()
    ########################
    ## Start training Paras Selection model
    ########################
    logging.info("2. Start training ParasSelection.")

    paras_selection.trigger_train()


    ########################
    ## Start training Paras Selection model
    ########################
    logging.info("2. Start inferring.")

    paras_selection.trigger_inference()
