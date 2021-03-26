# CHANGELOG

### This file contains changes and modifications during developing.

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