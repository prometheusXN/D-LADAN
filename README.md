# D-LADAN (under construction) 
The source code of article "**Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledge of Law Articles**", which is the revised journal version of article ''*Distinguish Confusing Law Articles for Legal Judgment Prediction*'', ACL 2020.

In this version, we consider the **confusing law article (or charge) problem** from both prior and posterior perspectives.
Through the analysis of the problem, we conflate the posterior confusing law article (or charge) problem and the data imbalance problem together and make further improvements to our conference model, [**LADAN**](https://github.com/prometheusXN/LADAN).

Compared with *LADAN*, our main improvement in this work is the proposed momentum-updated **revised memory mechanism**, which dynamically senses the posterior similarity relationships between law articles (or charges) learned by the model during the training process. Besides, a combined **weighted graph distillation operation** (GDO) is proposed to adaptively capture the distinguishable features from such a post-hoc similarity graph structure. So far, **D-LADAN** can correct the negative bias cuased by the data imbalance problem and accurately extract the differences between law articles (or charges) to distinguish them. Overall framework of D-LADAN is as follow:

![image](https://github.com/prometheusXN/D-LADAN/blob/main/fig/Framework%20of%20D_Ladan.jpg)

## Introduction
We refactored [LADAN](https://github.com/prometheusXN/LADAN) based on the **Tensorflow2.x**, and built this project.
Here, we briefly described the structure of the project and the functionality of each subfolder, as follow:

```
Config			// [important] the config of our D-LADAN and some baselines
Model_component     // [important] the model components that make up the various models.
├── Ladan_component.py      //[important] the main component of LADAN model.
├── Ladan_ppk_component_Criminal.py     // main LADAN component for the criminal datasets.
├── Ladan_ppk_component.py      //[important] the main component of of the D-LADAN model.
└── Ladanppk_component_Criminal.py 
Model     // [important] the keras model for various models.
├── LADAN_model.py      //[important] the Tensorflow 2.x version of the full LADAN+MTL model.
├── LADAN_model_C.py
├── DLADAN_model.py     //[important] the Tensorflow 2.x version of the full D-LADAN(+MTL, TOPJUDGE, MPBFN) model.
└── DLADAN_model_C.py
train     // [important] the train code for the corresponding models.
├── train_LADAN.py      // [important] the training code of LADAN+MTL model.
├── train_LADAN_C.py
├── train_DLADAN.py      // [important] the training code of D-LADAN model.
└── train_DLADAN_C.py 

utils     // some util functions used in our opject.

```

## Data Processing

## Usage
Basic requirements for all the libraries are in the `requirements.txt`. 

