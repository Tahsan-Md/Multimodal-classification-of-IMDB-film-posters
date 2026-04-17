# Multimodal Classification of IMDB Film Posters and Overviews

> Coursework 1 — Deep Learning module. A residual CNN and a bidirectional LSTM independently classify 6,325 films by genre using their posters and overview captions respectively, on a 25-way multi-label task.

---

## Overview

This project evaluates **two independently trained neural networks** on the [Multimodal IMDB dataset](https://arxiv.org/abs/1702.01992):

- a **CNN** with residual blocks that classifies films from their **posters** (64×64 RGB images), and
- a **bidirectional LSTM** that classifies films from their **overviews** (short text captions).

Both target the same 25-way multi-label genre problem, so their errors are directly comparable. The study reports per-genre F1, top-1 confusion matrices, a bucket analysis of the complementary error sets, and worked case studies with real film IDs.

## Key results

| Model | Best val-loss | Best epoch | Micro-F1 | Macro-F1 |
|:------|:-------------:|:----------:|:--------:|:--------:|
| CNN   (poster → genre)    | 0.2316 | 50 / 50 | 0.347 | 0.040 |
| **LSTM** (overview → genre) | **0.2118** | **5 / 20** | **0.391** | **0.061** |

The LSTM is the stronger unimodal model, converging in one-tenth the number of epochs. However, the two models err differently — the CNN alone succeeds on 179 films where the LSTM fails, and vice-versa on 223 films — so their error sets are complementary, the condition under which multimodal fusion would help.

## Repository structure

```
├── Keras_Assignment__2025_2026_B_.ipynb   # Completed notebook (training + evaluation)
├── Keras_Assignment_Report.docx           # 4-page critical analysis report
├── Keras_Assignment_Report.pdf            # PDF rendering of the report
├── Multimodal_IMDB_dataset/               # (not committed — see "Dataset" below)
│   ├── Images/                            # 7,896 poster JPEGs
│   └── IMDB_overview_genres.csv           # labels + captions
└── README.md
```

## Dataset

The Multimodal IMDB dataset ([Arevalo et al., 2017](https://arxiv.org/abs/1702.01992)) contains 6,325 films from the Internet Movie Database, each with:

- a colour **poster** in JPEG format (variable resolution),
- a short plain-text **overview**, and
- multi-hot labels across 25 candidate genres (21 actually present).

### Label distribution

The distribution is severely skewed, which dominates every modelling result:

| Genre | Count | Share | Tier |
|:------|------:|------:|:-----|
| Drama    | 3,422 | 54.1% | **Majority** |
| Comedy   | 2,754 | 43.5% | **Majority** |
| Action   | 1,295 | 20.5% | Frequent |
| Romance  | 1,293 | 20.4% | Frequent |
| Horror   |   484 |  7.7% | Mid |
| Sci-Fi   |   388 |  6.1% | Mid |
| War      |   116 |  1.8% | Rare |
| Musical  |    88 |  1.4% | Rare |
| Western  |    49 |  0.8% | Rare |

Four nominal classes (*Documentary, News, Reality-TV, Short*) never appear in the labels at all.

## Architectures

### CNN (658,873 parameters)

```
Input (64×64×3)
 └─ StemConv (3×3, 32 filters, ReLU)
 └─ ResidualBlock ×4   (32 → 64 → 128 → 128 filters)
 └─ MaxPool2D between blocks 1, 2, 3
 └─ GlobalAveragePooling2D
 └─ Dense(256, ReLU) + Dropout(0.5)
 └─ Dense(25, sigmoid)
```

Each `ResidualBlock` has a main path of two 3×3 convolutions (with dropout) and a 1×1 skip convolution. `GlobalAveragePooling2D` is used instead of `Flatten` as an implicit regulariser.

### LSTM (4,411,561 parameters)

```
TextVectorization (vocab 10,000)
 └─ Embedding(10000, 265, mask_zero=True)
 └─ Bidirectional LSTM(256, return_sequences=True, dropout 0.5, rec-dropout 0.2)
 └─ Bidirectional LSTM(128, dropout 0.5, rec-dropout 0.2)
 └─ Dense(128, ReLU) + Dropout(0.8)
 └─ Dense(25, sigmoid)
```

The embedding matrix alone accounts for 2.65 M of the total parameters.

## Quick start

### Prerequisites

- Python 3.10+
- A Google Colab account with GPU runtime (recommended — CPU training is prohibitively slow)
- ~200 MB free space on Google Drive for the unzipped dataset

### Running the notebook

1. Upload the unzipped `Multimodal_IMDB_dataset/` folder to the root of your Google Drive.
2. Open `Keras_Assignment__2025_2026_B_.ipynb` in Colab.
3. Set the runtime to GPU: **Runtime → Change runtime type → GPU**.
4. **Runtime → Run all.**

Full training takes roughly 30–60 minutes on a T4:

- CNN: 50 epochs × ~5 s/epoch ≈ 4–5 minutes
- LSTM: 20 epochs × ~120 s/epoch ≈ 40 minutes

The notebook saves best-validation-loss checkpoints for both models and reloads them before evaluation.

## What the notebook does

1. **Data processing** — an optimised `tf.data` pipeline with `AUTOTUNE` parallel mapping, shuffle, batching and prefetching for both the image and text streams.
2. **Model definition** — the `ResidualBlock` custom layer, the CNN via the Keras Functional API, and the LSTM via `Sequential`.
3. **Training** — 50 epochs for the CNN, 20 for the LSTM, with `ModelCheckpoint` saving the best weights.
4. **Evaluation** — training curves, per-genre F1 bars, top-1 confusion matrices, a bucket analysis (both right / only-CNN / only-LSTM / both wrong), and worked case studies with real film IDs.

## Findings in one paragraph

The CNN captures the **visual prior** of a genre (poster palette, composition); the LSTM captures its **lexical prior** (a single content word like *detective*, *alien*, *heist* can almost uniquely identify a class). Both models fall back on the Drama/Comedy majority classes when neither prior is strong. The per-genre F1 collapses to zero on classes with fewer than 100 positive examples — a known pathology of binary cross-entropy on imbalanced multi-label data, and the primary ceiling on generalisation. See `Keras_Assignment_Report.pdf` for the full critical analysis.

## Recommended next steps

- **Class-weighted BCE loss** to pull both models off the "predict-zero" attractor for rare classes.
- **Transfer learning** for the CNN from an ImageNet backbone (e.g. MobileNetV2) rather than training visual features from scratch on 5,060 posters.
- **Pre-trained word embeddings** (GloVe, fastText) to replace the 10,000 × 265 randomly-initialised embedding.
- **Late-fusion multimodal classifier** — the CNN and LSTM error sets are complementary, so a fusion model should beat either unimodal baseline.

## References

1. Arevalo, J., Solorio, T., Montes-y-Gómez, M., González, F. A. (2017). Gated multimodal units for information fusion. *arXiv:1702.01992*.
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. *Proc. IEEE CVPR*, 770–778.
3. Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Baltrušaitis, T., Ahuja, C., Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE TPAMI*, 41(2), 423–443.

## Author

- **Name:** _Your Name_
- **Student ID:** _Your ID_
- **Module:** Deep Learning — CW1
- **Submitted:** April 2026

## License

Academic coursework submission. The Multimodal IMDB dataset retains its original licence (see the [source repository](https://github.com/johnarevalo/gmu-mmimdb)).
