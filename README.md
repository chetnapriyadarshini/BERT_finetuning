# BERT Fine-Tuning for Sentence Pair Classification

A Jupyter Notebook demonstrating the fine-tuning of a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for sentence pair classification tasks, such as Natural Language Inference (NLI) or Semantic Textual Similarity (STS).

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Notebook Contents](#notebook-contents)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

---

## Overview

This project explores transfer learning in Natural Language Processing (NLP) by fine-tuning Google's BERT model on a downstream sentence pair classification task. The notebook walks through data preprocessing, tokenisation using the BERT tokenizer, model configuration, training, and evaluation.

---

## Background

BERT (Devlin et al., 2018) is a transformer-based language model pre-trained on large text corpora using masked language modelling and next sentence prediction objectives. Fine-tuning BERT on labelled task-specific data has become a standard paradigm in NLP, consistently achieving state-of-the-art results across a broad range of benchmarks.

Sentence pair classification involves determining the relationship between two input sentences — for example, whether one sentence entails, contradicts, or is neutral with respect to the other (as in the Stanford NLI dataset).

---

## Notebook Contents

| Section | Description |
|---|---|
| Data Loading & Exploration | Loading and inspecting the sentence pair dataset |
| Preprocessing | Tokenisation using `BertTokenizer`, padding, and attention mask generation |
| Model Configuration | Loading `BertForSequenceClassification` with task-specific classification head |
| Training | Fine-tuning with AdamW optimiser and learning rate scheduling |
| Evaluation | Computing accuracy and loss on the validation set |
| Inference | Running predictions on custom sentence pairs |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `transformers` (Hugging Face) | Pre-trained BERT model and tokenizer |
| `torch` (PyTorch) | Deep learning framework |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` | Training curve visualisation |

---

## Setup and Installation

```bash
git clone https://github.com/chetnapriyadarshini/BERT_finetuning.git
cd BERT_finetuning
pip install transformers torch pandas numpy scikit-learn matplotlib
```

Launch the notebook:

```bash
jupyter notebook Fine_tuning_BERT_for_Sentence_Pair_Classification.ipynb
```

---

## Usage

Open the notebook and execute the cells sequentially. Ensure a GPU runtime is available (e.g., Google Colab with a T4 or A100 GPU) for acceptable training times. The notebook is self-contained and includes inline explanations for each step.

---

## Results

The fine-tuned BERT model achieves competitive classification accuracy on the validation set. Detailed training and validation metrics are reported within the notebook.

---

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
