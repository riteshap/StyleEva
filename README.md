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
[ADD THE EXACT DATASET URL USED IN THIS PROJECT]

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
