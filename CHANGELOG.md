# CHANGELOG

### This file contains changes and modifications during developing.

## May 6 2021, 22:44
- Reorganize *configs.py*
- Inferring in Transformer is now totally independent flow
- Temporarily remove calculating metrics, just generating predicted answers only
- Add 2 more fields `ans1_plain` and `ans1_plain` as creating iterator. Those are for inferring only.
- Replace user-generated vocab by BERT vocab for creating data for training/eval/test.

## May 2 2021, 21:18
- Reimplement *BeamSearch* by simplifiyng many steps
- Switching context now occurs forcibly after training an epoch
- Process in module "DataReading" now runs on single process
- Function *build_vocab* adds special tokens to vocab file


## Apr 25 2021, 16:18
- Reimplement function calculating *ROUGE-L*, *BLEU-1*...
- Implement *Beam Search*
- Implement new answer inferring module namely *TransDecoder* ans use *Beam Search*
- Add feature logging metric calculation in testing phase

## Apr 21 2021, 15:07
- Fix errors in processing *movie* context in *DataReading* module
- Reconstruct *DataReading* module
- Replace **LSTM-based** by **BERT-based** *Embedding layer*
- Change arguments of *PGD's forward* to fit with new *Embedding layer*
- Modify training process's variables in model to fit with new *Embedding layer*
- Reorganize *GraphReasoning* module into new folder
- Use trick longer answer to calculate loss and 2 answers to calculate metrics

## Apr 19 2021, 10:53
- Change and reconstruct repo to fit with *GCN* setting

## Apr 6 2021, 14:45
- Apply parallel processing and shared memory to *DataReading* module
- Back up processed context file => speed up processing time

## Apr 6, 2021, 14:45
- Change module *data_reading* significantly:
    - Construct entire module into class, class *DataReading*
    - Define may to process story and movie script separatedly
    - Process context before process each entry => speed up processsing
    - Apply top-down golden paras selection instead of tree-based

## Mar 30, 2021, 16:35
- Change minor thing in *IAL* to speed up training
- Fix some problems in *trigger_inference*

## Mar 28, 2021, 22:17
- Reduce vocab size to 10 times smaller: Use top 1000 most occured words from context and words from answer
- Change in accumulating loss (detaching it before accumulating)
- Contiguous and change dimensions at some points in *PGD*
- Change the way of calculating *Equation 2*, hence reduce time and memory prohibitively spent during training

## Mar 26, 2021, 20:35
- Reimplement *Dataset* module
- Finish test a training batch

## Mar 25, 2021, 11:18
- Finish remaining modules for training, testing and evaluating.

## Mar 19, 2021, 18:05
- Finish implementing *PointerGeneratorDecoder*
- Add question tokens to global vocab of *PointerGeneratorDecoder*

## Mar 16, 2021, 12:54
- Finish module *FineGrain*, *CustomDataset*, *IAL*

## Mar1, 2021, 16:50
- Finish *FineGrain* module with single question and paragraph as input
- Modify *config* file

## Mar 8, 2021, 12:00
- Abandon idea of using trained Bert-based Paras Selector and back to idea proposed in paper *Simple and Effective* - using TfIdf

## Mar 2, 2021
- Use *csv* format instead of *pickle* for data backup
- Finish implementing training and inferring module for *ParasSelection*

## Feb 21, 2022
- Use 'DataFrame' in *pandas* to pickle data and for easily deal with 'Dataset' of *huggingface*
- Reformat the data to be backed up

## Feb 17, 2021
- Reorder resources of paths of backup file. From now, paths are stored in `configs.py`

## Feb 16 2021, 15:34 PM
- Finish 'data_reading' module.

## Feb 13 2021, 10:18 AM
- Add 'data_reading' module to read, initially preprocess and decompose documents into paragraphs

## Feb 3 2021, 11:00 AM
- Very first commit
- Add default files: `.gitignore`, `CHANGELOG.md`, `configs.py`, `launcher.sh` and `README.md`