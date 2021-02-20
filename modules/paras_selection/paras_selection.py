
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


    ## Check whether total files are created.
    ## Since train, valid and test dataset cannot fit into memory,
    ## I read part of each and save to individual file.
    ## Therefore, assume I split each dataset into 8 (args.n_shards) parts,
    ## the total number of files creates are 3 * 8 = 24.
    n_files = 0
    for ith in range(3 * args.n_shards):
        if check_file_existence(PATH['golden_paras'].replace("[N_PART]", str(ith))):
            n_files += 1
        else:
            break

    if n_files < 3 * args.n_shards:
        GoldenParas().generate_goldenParas(n_files)
    else:
        logging.info("=> Golden paras files are created sufficiently.")



    ########################
    ## Start training Paras Selection model
    ########################
    ## TODO: Implement: call ParasSelection and trigger training
    # logging.info("2. Start training ParasSelection.")

    # paras_selection = ParasSelection()