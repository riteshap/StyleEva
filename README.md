# StyleEva

Code and data for the study:

**Modeling Paradigms for Relational Discrimination in Dense Stylistic Hierarchies**

## 1. Description

This repository contains the implementation and experimental data used in our study of dense stylistic intensity discrimination.

The study constructs source-conditioned dense stylistic hierarchies from multiple intermediate text variants and evaluates three representative modeling paradigms:

1. Generic Embedding
2. Multiclass Classification
3. Pairwise Relational Learning

The repository includes the generated intermediate texts, LLM-based stylistic intensity scores, hierarchy construction code, and the implementations used to evaluate the three modeling paradigms.

The stored intermediate texts and scoring results can be used directly for reproducing the main experiments without regenerating all LLM outputs.

---

## 2. Dataset Information

Experiments are conducted on two text style transfer datasets:

### Yelp Sentiment

The Yelp Sentiment dataset contains negative and positive review texts commonly used for sentiment style transfer.

Source:
https://github.com/passeul/style-transfer-model-evaluation

In this study, negative reviews are used as source texts and positive sentiment is treated as the target style.

### Shakespeare

The Shakespeare dataset contains parallel Shakespearean-style and modern English sentences and is used for literary style transfer.

Original source:
https://github.com/harsh19/Shakespearizing-Modern-English  

In this study, Shakespearean-style sentences are used as source texts and modern English is treated as the target style.

### Data used in this repository

For each dataset:

- 1,000 source texts are used for training.
- 100 source texts are used for testing.
- Multiple intermediate stylistic variants are generated for each source text.
- Duplicate candidates are removed.
- Candidates with source-text semantic similarity below 0.60 are removed.

The processed candidate texts used in the experiments are already included in this repository.

---

## 3. Repository Structure

The main files and directories are organized as follows:

```text
StyleEva/
│
├── get_cand/
│   └── result/
│       └── Generated intermediate stylistic candidates
│
├── get_score/
│   ├── score/
│   │   └── LLM-based stylistic intensity scoring results
│   └── sbert_model.py
│       └── Generic Embedding experiments
│
├── multi_class/
│   └── model_class.py
│       └── Multiclass Classification experiments
│
├── contrastive/
│   └── model_class.py
│       └── Pairwise Relational Learning experiments
│
└── README.md
```
## 4. Usage Instructions

The main experimental scripts are organized according to the three comparison models described above. In all three model directories, task1 corresponds to the Yelp Sentiment dataset, while task8 corresponds to the Shakespeare dataset.

For Generic Embedding, run:
```text
/get_score/sbert_model.py
```
Before execution, specify the project root directory and the local path of the all-mpnet-base-v2 model.

For Multiclass Classification, run:
```text
/multi_class/model_class.py
```
Before execution, specify the project root directory and the local path of the Llama-3.2-3B-Instruct model used for token representation extraction.

For Pairwise Relational Learning, run:
```text
/contrastive/model_class.py
```
Similarly, specify the project root directory and the local path of Llama-3.2-3B-Instruct before execution.

The scripts under /get_cand and /get_score are used to request LLMs for generating dense intermediate stylistic variants and obtaining repeated stylistic intensity scores. Since all intermediate texts and scoring results used in the experiments are already provided in this repository, these scripts do not need to be rerun for reproducing the main experiments. If regeneration is required, users must provide their own corresponding LLM API keys.

## 5. Requirements

The main Python dependencies used in this project are:

```text
torch
numpy
tqdm
transformers
sklearn
sentence_transformers
```
The scripts used for external LLM requests additionally require:
```text
openai
```
