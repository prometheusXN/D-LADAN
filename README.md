# D-LADAN (under construction) 
The source code of the article "**Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledge of Law Articles**", TOIS. It is the revised journal version of the article ''*Distinguish Confusing Law Articles for Legal Judgment Prediction*'', ACL 2020.

This version considers the **confusing law article (or charge) problem** from both prior and posterior perspectives.
Through the analysis of the problem, we conflate the posterior confusing law article (or charge) problem and the data imbalance problem together and make further improvements to our conference model: [**LADAN**](https://github.com/prometheusXN/LADAN).

Compared with *LADAN*, our main improvement in this work is the proposed momentum-updated **revised memory mechanism**, which dynamically senses the posterior similarity relationships between law articles (or charges) learned by the model during the training process. Besides, a combined **weighted graph distillation operation** (GDO) is proposed to adaptively capture the distinguishable features from such a post-hoc similarity graph structure. So far, **D-LADAN** can correct the negative bias caused by the data imbalance problem and accurately extract the differences between law articles (or charges) to distinguish them. The overall framework of D-LADAN is as follows:

![image](https://github.com/prometheusXN/D-LADAN/blob/main/fig/Framework%20of%20D_Ladan.jpg)

## Introduction
We refactored [LADAN](https://github.com/prometheusXN/LADAN) based on the **Tensorflow2.x**, and built this project.

* new word2vec files: [cail_thulac_new](https://drive.google.com/file/d/1UJZUplgxeIGlLoiRz_iFzRtcTW-86TPJ/view?usp=sharing)


Here, we briefly describe the structure of the project and the functionality of each subfolder, as follows:

```
Config			// [important] the config of our D-LADAN and some baselines
Model_component     // [important] the components that comprise the various models.
├── Ladan_component.py      //[important] the main component of the LADAN model.
├── Ladan_ppk_component_Criminal.py     // main LADAN component for the criminal datasets.
├── Ladan_ppk_component.py      //[important] the main component of the D-LADAN model.
└── Ladanppk_component_Criminal.py 
Model     // [important] the keras model for various models.
├── LADAN_model.py      //[important] the Tensorflow 2.x version of the full LADAN+MTL model.
├── LADAN_model_C.py
├── DLADAN_model.py     //[important] the Tensorflow 2.x version of the full D-LADAN(+MTL, TOPJUDGE, MPBFN) model.
└── DLADAN_model_C.py
train     // [important] the train code for the corresponding models.
├── train_LADAN.py      // [important] the training code of the LADAN+MTL model.
├── train_LADAN_C.py
├── train_DLADAN.py      // [important] the training code of the D-LADAN model.
└── train_DLADAN_C.py 

utils     // some util functions used in our project.

```

## Data Processing
For training, you should generate the input data format and put them into the "__processed_dataset__" folder.

You can download our processed dataset from our Google Driver: [**processed dataset (for D-LADAN)**](https://drive.google.com/file/d/1-YRQ0bVok62ToHX2Fu0y8QIDygWxSbQ8/view?usp=share_link).

## Usage
The basic requirements for all the libraries are in the `requirements.txt`. 

## Future Work
There are many recent works comparing the models that have a PLM backbone with our work to verify their effectiveness.
We complain that such a comparison is unfair and does not directly prove the superiority of their frameworks, due to the inherent performance differences between the simple backbones (e.g., CNN, RNN, and LSTM) and the PLM-like backbone (e.g., BERT, Lawformer, and so on).
Thus, we'll implement BERT versions of LADAN and D-LADAN later.
We call for an objective and fair comparison between methods, to further advance the development of the industry.

## Bert_Ladan
Now we have implemented BERT versions of D-LADAN and LADAN, refer to the folder: [**Bert_Ladan**](https://github.com/prometheusXN/D-LADAN/tree/main/Bert_Ladan).
