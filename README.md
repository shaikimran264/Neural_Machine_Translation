# English → Hindi Neural Machine Translation

A Transformer-based sequence-to-sequence model for English-to-Hindi translation, trained **from scratch** (no pretrained weights).

---

## Results

| Metric   | Score  |
|----------|--------|
| BLEU     | 9.45   |
| chrF     | 31.48  |
| ROUGE-1  | 40.91  |
| ROUGE-2  | 15.23  |
| ROUGE-L  | 35.31  |

---

## Evaluation Metrics — Formulas

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision between the hypothesis (predicted) and reference (ground truth) translation, with a brevity penalty for short outputs.

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ = modified n-gram precision for n-grams of order $n$
- $w_n = \frac{1}{N}$ (uniform weights, $N = 4$)
- Brevity Penalty: $BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$
- $c$ = length of the hypothesis, $r$ = length of the reference

$$p_n = \frac{\sum_{\text{ngram} \in \hat{y}} \min\!\left(\text{count}(\text{ngram}, \hat{y}),\ \text{count}(\text{ngram}, y)\right)}{\sum_{\text{ngram} \in \hat{y}} \text{count}(\text{ngram}, \hat{y})}$$

> A BLEU score of **9.45** is typical for a from-scratch Transformer on a morphologically rich target language like Hindi with limited data.

---

### chrF (Character n-gram F-score)

Computes F-score at the **character** level rather than word level — better suited for morphologically rich languages like Hindi.

$$\text{chrF} = \frac{(1 + \beta^2) \cdot \text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

Where ($\beta = 2$ by default in `sacrebleu`):

$$\text{chrP} = \frac{\text{matched char n-grams in hypothesis}}{\text{total char n-grams in hypothesis}}$$

$$\text{chrR} = \frac{\text{matched char n-grams in hypothesis}}{\text{total char n-grams in reference}}$$

> chrF of **31.48** is more informative than BLEU for Hindi because it rewards partial word matches (e.g., correct verb stems).

---

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Originally designed for summarization; measures overlap between hypothesis and reference.

#### ROUGE-1 and ROUGE-2 (unigram / bigram overlap)

$$\text{ROUGE-N} = \frac{\sum_{\text{ngram} \in y} \min\!\left(\text{count}(\text{ngram}, \hat{y}),\ \text{count}(\text{ngram}, y)\right)}{\sum_{\text{ngram} \in y} \text{count}(\text{ngram}, y)}$$

Where $N=1$ for ROUGE-1 (unigrams) and $N=2$ for ROUGE-2 (bigrams).

The **F1 version** (used here) balances precision and recall:

$$\text{ROUGE-N}_{F1} = \frac{2 \cdot P \cdot R}{P + R}$$

#### ROUGE-L (Longest Common Subsequence)

Measures the longest common subsequence (LCS) between hypothesis $\hat{y}$ and reference $y$, rewarding fluency and in-order matches.

$$R_{lcs} = \frac{|\text{LCS}(\hat{y},\ y)|}{|y|}, \quad P_{lcs} = \frac{|\text{LCS}(\hat{y},\ y)|}{|\hat{y}|}$$

$$\text{ROUGE-L}_{F1} = \frac{(1+\beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

| Metric   | What it captures |
|----------|-----------------|
| ROUGE-1  | Unigram (word) overlap |
| ROUGE-2  | Bigram (two-word phrase) overlap |
| ROUGE-L  | Longest in-order word sequence |

---

## Model Architecture

| Hyperparameter       | Value |
|----------------------|-------|
| Architecture         | Transformer (encoder-decoder) |
| `d_model`            | 300 |
| Attention heads      | 6 |
| Encoder layers       | 6 |
| Decoder layers       | 6 |
| Dropout              | 0.1 |
| Max sequence length  | 50 (both languages) |

### Embeddings

| Language         | Embedding                         | Dim |
|------------------|-----------------------------------|-----|
| English (source) | GloVe 6B 300d                     | 300 |
| Hindi (target)   | FastText `cc.hi.300.bin` (v0.9.3) | 300 |

Embeddings are **fine-tuned** during training (not frozen).

---

## Training Configuration

| Setting         | Value |
|-----------------|-------|
| Loss function   | `CrossEntropyLoss` (ignores `<PAD>`) |
| Optimizer       | Adam |
| Learning rate   | Dynamic — `ReduceLROnPlateau(mode="min", factor=0.5, patience=3)` |
| Batch size      | 32 |
| Epochs          | 10 |
| Mixed precision | AMP (`autocast` + `GradScaler`) |

---

## Tokenization

| Language | Tokenizer |
|----------|-----------|
| English  | spaCy `en_core_web_sm` v3.8.11 |
| Hindi    | `indic-nlp-library` `trivial_tokenize_indic` v0.92 |

---

## Project Structure

```
├── main.ipynb            # Full pipeline: preprocessing → training → evaluation
├── test.ipynb            # Inference + evaluation only (loads saved checkpoint)
├── requirements.txt      # Python dependencies
└── README.md
```

> Required files (dataset, embeddings, tokenized pickles, model checkpoint) are hosted on Google Drive — see **Required Files** below.

---

## Setup & Reproduction

### 1. Clone the repo

```bash
git clone https://github.com/shaikimran264/Neural_Machine_Translation
cd Neural_Machine_Translation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. Download external embedding files

| File | Source | Size |
|------|--------|------|
| `cc.hi.300.bin` | [FastText Hindi vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz) | ~4 GB |
| `glove.6B.300d.txt` | [GloVe 6B](https://nlp.stanford.edu/data/glove.6B.zip) | ~1 GB |

```bash
# Hindi FastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz
gunzip cc.hi.300.bin.gz

# GloVe
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip   # use glove.6B.300d.txt
```

### 5. Download required project files

All required files (dataset JSON, tokenized pickle files, trained model checkpoint) are available at:

**[Google Drive – Required Files](https://drive.google.com/file/d/1pvP1AB7Wgj89gkEWPQ762GmvuPTsgH8d/view?usp=sharing)**

---

## How to Run

### Train + evaluate (reproduce all results):
```bash
jupyter notebook code.ipynb
```
Run all cells top to bottom. Load required files from the Drive link above.

### Evaluate only (load saved checkpoint, skip training):
```bash
jupyter notebook test.ipynb
```
Load required files from the same Drive link above.
