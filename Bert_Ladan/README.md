#  The Upgraded Version: D-LADAN meets Transformers
To fairly compare with the related works involving the pre-train language mode(PLM) backbones, such as BERT, Lawformer, etc, we propose a Transformer version of LADAN and D-LADAN to verify the effectiveness of the model strategy.

## Introduction
We constructed a Transformer version of D-LADAN and LADAN based on the **Pytorch** and built this project here.

Here, we briefly describe the structure of the project and the functionality of some important subfolder, as follows:

```
load_dataset    // [important] the code to generate the data form.
├── CAIL_dataset_full_chinese.py  // the code for CAIL dataset.
├── Criminal_dataset.py    // the code for Criminal dataset.
Formatters    // [important] Convert the data to the input format of model.
├── SentenceLevelFormatter.py
Dataloaders    // [important] the Dataloader of torch
├── DLADAN_CAIL_loader.py
├── DLADAN_Criminal_loader.py
DLADAN_bert.py ──├── The whole framework of DLADAN (and LADAN) model.
LADAN_bert.py  ──├ 
Dladan_component.py ──├── Body part of the corresponding model.
Ladan_component.py  ──├ 
train_DLADAN_C.py  ──────────────────├── The code of model training.
train_DLADAN_full_chinese.py       ──├
train_DLADAN_full_chinese_large.py ──├
