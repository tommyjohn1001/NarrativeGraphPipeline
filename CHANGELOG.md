# CHANGELOG

### This file contains changes and modifications during developing.

## Apr 19 2021, 11:11
- Update changes in *PGN* of branch *dev-proposal* to current branch

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