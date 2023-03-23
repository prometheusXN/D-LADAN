# D-LADAN (under construction) 
The source code of article "**Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledge of Law Articles**" in TOIS, which is the revised journal version of article ''*Distinguish Confusing Law Articles for Legal Judgment Prediction*'', ACL 2020.

In this version, we consider the **confusing law article (or charge) problem** from both prior and posterior perspectives.
Through the analysis of the problem, we conflate the posterior confusing law article (or charge) problem and the data imbalance problem together and make further improvements to our conference model, [**LADAN**](https://github.com/prometheusXN/LADAN).

Compared with *LADAN*, our main improvement in this work is the proposed momentum-updated **revised memory mechanism**, which dynamically senses the posterior similarity relationships between law articles (or charges) learned by the model during the training process. Besides, a combined **weighted graph distillation operation** (GDO) is proposed to adaptively capture the distinguishable features from such a post-hoc similarity graph structure. So far, **D-LADAN** can correct the negative bias cuased by the data imbalance problem and accurately extract the differences between law articles (or charges) to distinguish them. Overall framework of D-LADAN is as follow:

![image](https://github.com/prometheusXN/D-LADAN/blob/main/fig/Framework%20of%20D_Ladan.jpg)

## Introduction
We refactored [LADAN](https://github.com/prometheusXN/LADAN) based on the **Tensorflow2.x**, and built this project.
Here, we briefly described the structure of the project and the functionality of each subfolder, as follow:

```
Config			// [important] the config of our D-LADAN and some baselines
data
├── candidates                       
│   ├── candidates1.zip			// [important] candidate zipfile 1(for query 1-50)
│   └── candidates2.zip			// [important] candidate zipfile 2(for query 51-107)
├── corpus
│   ├── common_charge.json
│   ├── controversial_charge.json
│   └── document_path.json		// corpus document path file
├── label
│   └── label_top30.json		// [important] labels of top 30-relevant candidates, the rest unlabelled candidates are irrelevant (or label=0)
├── others
│   ├── criminal charges.txt		// list of all Chinese criminal charges
│   └── stopword.txt
├── prediction				// candidate pooling results using different methods
│   ├── bert.json
│   ├── bm25_top100.json
│   ├── combined_top100.json		// overall candidate list
│   ├── lm_top100.json
│   └── tfidf_top100.json
└── query
    └── query.json			// [important] overall query file
```

## Data Processing

## Usage
Basic requirements for all the libraries are in the `requirements.txt`. 

