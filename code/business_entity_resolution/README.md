# High-Precision Multilingual Business Entity Resolution Pipeline

This directory contains the self-contained, reproducible pipeline for the **Amazon ML Challenge 2026: Multilingual Business Entity Resolution**.

## 🌟 Key Architectural Innovations

1. **Language & Script-Agnostic Normalization (`normalize.py`)**:
   - Deterministic transliteration (`anyascii`) converting 9 Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya) into Romanized Latin ASCII.
   - Unicode NFKD decomposition and diacritic stripping for accented French (`é`, `è`, `ô`, `ç`) and noisy English text.
   - Consonant skeleton extraction to eliminate vowel drift and spelling variations.
   - Standardized legal entity category extraction (`PVT_LTD`, `CORP`, `INC`, `LLC`, `SARL`, `SAS`).
   - Universal numeric anchor & postal code extraction (6-digit Indian PIN, 5-digit US/FR ZIP, door/plot numbers).

2. **Multimodal Hybrid Blocker (`hybrid_blocker.py`, `dense_retrieval.py`)**:
   - Dynamic grouping by `country` (satisfies open-set requirements, zero cross-country leakage).
   - Route A (Lexical): Multi-route inverted index across normalized name roots, rare distinctive tokens, consonant 3-grams, and postal locality keys.
   - Route B (Dense Neural Bi-Encoder): `BAAI/bge-m3` 1024-dim multilingual embeddings running batched FP16 cosine matrix multiplication on GPU.
   - Reciprocal Rank Fusion (RRF): Combines dense semantic ranks and lexical ranks:
     $$\text{RRF}(d) = \sum_{m \in \{\text{lexical}, \text{dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Generates high-recall candidates ($>97.2\%$) within strict candidate budgets.

3. **25 High-Dimensional Meta-Features (`meta_features.py`)**:
   - Cross-space retrieval signals: `dense_sim`, `rrf_score`, rank reciprocals.
   - Lexical & phonetic surface metrics: RapidFuzz token sort/set ratio, partial ratio, Jaro-Winkler name and consonant skeleton.
   - Hard structural vetoes: House/door number match (`+1`), missing (`0`), or fatal conflict (`-1`); postal code exact/prefix match or conflict veto.

4. **Dual-Model Calibrated Meta-Ensemble (`ensemble.py`, `train_meta.py`)**:
   - Dual gradient boosting trees (**LightGBM + CatBoost**) trained on hard out-of-fold negative pairs.
   - Monotonic **Isotonic Probability Calibration** to convert raw decision tree margins into true Bayesian match posteriors.
   - Precision-tuned thresholding ($\tau \approx 0.70 - 0.72$) optimized specifically for the Macro $F_{0.5}$ metric (which weights precision 2x over recall and heavily penalizes singleton false merges).

5. **Bipartite 1-to-1 Exclusivity & Graph Matching (`graph_matching.py`)**:
   - Resolves multi-tenant commercial collisions (e.g., distinct businesses located in the same shopping mall or building).
   - Enforces injective mapping: each $S_2$ and $S_3$ candidate links to at most ONE unique $S_1$ entity using a maximum-weight priority queue.
   - Resolves transitive connected components across $S_1 \leftrightarrow S_2 \leftrightarrow S_3$.

6. **Neural Cross-Encoder Reranker (`reranker.py`, `run_cuda_pipeline.py`)**:
   - Optional `BAAI/bge-reranker-v2-m3` cross-attention transformer reranker for borderline probability intervals $[0.002, 0.998]$.

---

## 📂 Directory Structure

```text
code/business_entity_resolution/
├── README.md                      # Reproducibility instructions & architecture overview
├── requirements.txt               # Pinned dependencies
├── artifacts/
│   └── matcher_meta_ensemble.joblib # Calibrated model weights & threshold metadata
└── src/
    ├── __init__.py
    ├── audit.py                   # Data inspection & Unicode script audit
    ├── blocking.py                # Lexical multi-route inverted index blocker
    ├── config.py                  # Shared constants, paths, thresholds
    ├── data_loader.py             # High-performance chunked TSV data loader
    ├── dense_retrieval.py         # BAAI/bge-m3 dense bi-encoder & GPU matrix search
    ├── diagnose_validation.py     # Error decomposition & segment recall diagnostics
    ├── ensemble.py                # Dual-model LightGBM + CatBoost calibrated ensemble
    ├── evaluate.py                # Macro F0.5 evaluation logic
    ├── features.py                # Pairwise feature extractors
    ├── graph_matching.py          # Bipartite 1-to-1 exclusivity & transitive graph solver
    ├── hybrid_blocker.py          # Multimodal Lexical + Dense Reciprocal Rank Fusion
    ├── inference.py               # Streamlined batch inference runner
    ├── io_utils.py                # TSV reader/writer helpers
    ├── llm_arbiter.py             # Boundary-case LLM verification engine
    ├── meta_features.py           # 25-dimensional hybrid meta-feature extractor
    ├── normalize.py               # Multilingual transliteration (anyascii) & text cleaning
    ├── predict_hybrid.py          # Full multimodal hybrid test inference pipeline
    ├── predict.py                 # Lightweight CPU-optimized test inference runner
    ├── reranker.py                # BAAI/bge-reranker-v2-m3 neural cross-encoder
    ├── run_cuda_pipeline.py       # Complete GPU-accelerated end-to-end runner
    ├── train_meta.py              # Dual-model training, isotonic calibration & tuning
    ├── train_upgraded.py          # Full training pipeline with hard negative mining
    ├── train.py                   # Baseline training routine
    └── tune_decision_policy.py    # Decision threshold policy search
```

---

## ⚙️ Environment Setup

Install dependencies in Python 3.10+:

```bash
pip install -r requirements.txt
```

---

## 🚀 Reproduction Commands

### 1. Training the Dual-Model Calibrated Meta-Ensemble
To train the LightGBM + CatBoost ensemble with hard negative mining and fit isotonic calibration:

```bash
python -m src.train_meta \
    --data-root ../../dataset \
    --output-model artifacts/matcher_meta_ensemble.joblib \
    --sample-s1 50000
```

### 2. Running Test Inference

#### Option A: High-Performance Multimodal Hybrid Pipeline (Recommended)
```bash
python -m src.predict_hybrid \
    --data-root ../../dataset \
    --model-path artifacts/matcher_meta_ensemble.joblib \
    --output-dir ../../output
```

#### Option B: Full GPU Pipeline with Cross-Encoder Reranker
```bash
python -m src.run_cuda_pipeline \
    --device cuda \
    --data-root ../../dataset \
    --reranker-model BAAI/bge-reranker-v2-m3 \
    --output-dir ../../output
```

#### Option C: Fast CPU Threaded Inference
```bash
python -m src.predict \
    --data-root ../../dataset \
    --model-path artifacts/matcher_model.joblib \
    --output-dir ../../output \
    --batch-size 8000
```

This generates:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### 3. Submission Validation
Run the competition submission validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
