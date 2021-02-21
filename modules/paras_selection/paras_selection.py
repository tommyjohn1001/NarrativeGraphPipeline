
from modules.utils import load_object, save_object, check_file_existence
from modules.paras_selection.utils import GoldenParas
from configs import args, logging, PATH


if __name__ == '__main__':
    logging.info("* Paras Selection training.")

    ########################
    ## Create data for training
    ########################
    ## In this step, we use an heuristic method to filter
    ## golden paragraphs for each question, and use them
    ## to train ParasSelection model then
    logging.info("1. Create golden paras.")

    GoldenParas().generate_goldenParas()

    logging.info("=> Data files are created and saved sufficiently.")


    ########################
    ## Start training Paras Selection model
    ########################
    ## TODO: Implement: call ParasSelection and trigger training
    # logging.info("2. Start training ParasSelection.")

    # paras_selection = ParasSelection()