# 🏢 Multilingual Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Metric: Macro F0.5](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.9761-success.svg)](#-benchmark-results--ablation-studies)
[![Precision: 99.24%](https://img.shields.io/badge/Validation%20Precision-99.24%25-blueviolet.svg)](#-benchmark-results--ablation-studies)
[![Model: GBDT](https://img.shields.io/badge/Model-XGBoost%20%7C%20LightGBM-brightgreen.svg)](#-system-architecture)
[![Validator: PASS](https://img.shields.io/badge/Validator-PASS-success.svg)](#-submission-validation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **High-precision, ultra-scalable machine learning entity resolution pipeline** engineered for multilingual commercial record linkage under the asymmetric **Macro $F_{0.5}$** metric. Successfully resolves multi-source business records across $1.73\text{M}$ test entities and $11.5\text{M}+$ noisy vendor records with zero external API dependencies.

---

## 📌 Executive Summary & Key Insights

In enterprise commercial platforms, business identity data originates from disparate, noisy sources without shared global identifiers. The system evaluates record pairs under the official competition metric **Macro $F_{0.5}$**:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Core Constraints & Findings:
1. **Precision Weighted $2\times$ over Recall:** Falsely merging two distinct businesses incurs double the penalty of missing a true link.
2. **Singleton Sensitivity:** Approximately $14.2\%$ of Source 1 entities are singletons ($0$ matches). Correctly predicting an empty match list yields $1.0$, whereas even a single false-positive match drops the entity score to $0.0$.
3. **Zero Cross-Country Matching Law:** In an audit of $7.6\text{M}+$ ground-truth links, **$0.00\%$ cross national borders**. Partitioning candidate search strictly by country eliminates cross-border errors and slashes memory footprint to $<2.0\text{ GB RAM}$.
4. **Multilingual Script Divergence:** $100\%$ of Source 1 reference records are Romanized. In contrast, $\sim 18.9\%$ of Indian vendor records in Source 2 and 3 use native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya). Without phonetic transliteration, standard lexical string overlap is $0.00\%$.
5. **Universal Numeric & Building Anchors:** House/suite numbers (`RZ-142`, `1410`, `Suite 903`) and postal codes remain language-invariant. Primary house number conflicts between co-located businesses on the same street serve as a decisive veto against false merges.

---

## 🏗️ System Architecture

Our solution establishes a decoupled, multi-stage machine learning pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: NORMALIZATION                          │
│  - Unicode NFKD decomposition & diacritic stripping (é -> e, ô -> o)   │
│  - Multilingual script transliteration (Indic native scripts -> ASCII) │
│  - International legal suffix removal (Pvt Ltd, LLC, SARL, SAS, GmbH)  │
│  - Primary house/door number & postal code token extraction            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│        STAGE 2: MULTI-ATTRIBUTE INVERTED INDEX BLOCKING (96.5% RECALL) │
│  - Partition search space strictly by Country (US, India, France)      │
│  - 8 Composite indexing keys (compact name, core name, address number) │
│  - Frequency pruning (>150–300 count) for generic stop-words           │
│  - Reduces Cartesian search space by > 99.98% (~14.5 candidates/S1)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 3: 32-FEATURE PAIRWISE CLASSIFICATION            │
│  - RapidFuzz C++ kernels: Levenshtein, Jaro-Winkler, Token Set/Sort    │
│  - Character 3-gram Jaccard similarity & first-token conflict flags    │
│  - Primary house number match vs. mismatch indicators                  │
│  - Gradient Boosted Decision Trees (XGBoost / LightGBM)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         STAGE 4: DUAL-THRESHOLDING & 1-TO-1 TARGET EXCLUSIVITY         │
│  - Calibrated source-specific thresholds: τ_S2 = 0.70, τ_S3 = 0.75     │
│  - Global confidence priority queue for Source 2 and Source 3 targets  │
│  - Enforces 1-to-1 target exclusivity (each S2/S3 maps to at most 1 S1)│
│  - Generates byte-compliant matching_results.tsv & candidate_pairs.tsv │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results & Ablation Studies

### Iterative Validation Progression

| Experiment ID | Model / Architecture | Candidate Strategy | Feature Space | Global Exclusivity | Precision | Recall | Macro $F_{0.5}$ | False Positives | False Negatives |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | LightGBM Baseline | SuperBlocking Top-15 | 28 features | No ($\tau=0.50$) | 0.9815 | 0.9521 | 0.968730 | 312 | 834 |
| **EXP-02** | LightGBM + Thresh Tuning | SuperBlocking Top-15 | 28 features | No ($\tau=0.82$) | 0.9922 | 0.9394 | 0.973518 | 129 | 1056 |
| **EXP-03** | LightGBM + Exclusivity | SuperBlocking Top-15 | 28 features | Yes ($\tau=0.82$) | 0.9932 | 0.9394 | 0.974255 | 112 | 1056 |
| **EXP-04** | LightGBM + Src Thresholds | SuperBlocking Top-15 | 28 features | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.80$) | 0.9920 | 0.9425 | 0.974534 | 133 | 1002 |
| **EXP-05** | Ablation: Name Only | SuperBlocking Top-15 | 14 Name feats | Yes | 0.8859 | 0.8734 | 0.889718 | 1959 | 2206 |
| **EXP-06** | Ablation: Address Only | SuperBlocking Top-15 | 10 Addr feats | Yes | 0.9690 | 0.8494 | 0.919123 | 474 | 2624 |
| **EXP-07** | HistGradientBoosting | SuperBlocking Top-15 | 28 features | Yes | 0.9938 | 0.9391 | 0.974371 | 102 | 1061 |
| **EXP-08** | Deep LightGBM | SuperBlocking Top-15 | 28 features | Yes | 0.9914 | 0.9440 | 0.974164 | 143 | 976 |
| **EXP-09** | **Enhanced Feats + Top-20** | **SuperBlocking Top-20** | **32 features** | **Yes ($\tau_{S2}=0.75, \tau_{S3}=0.85$)** | **0.9924** | **0.9454** | **0.976105** | **126** | **951** |
| **EXP-10** | **Ensemble (LGB + XGB)** | **SuperBlocking Top-20** | **32 features** | **Yes** | **0.9932** | **0.9439** | **0.976653** | **113** | **977** |

---

## 📂 Repository Directory Layout

```text
Multilingual_Business_Entity_Resolution/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py           # Package initializer
│       │   ├── normalization.py      # Script-agnostic phonetic transliteration & legal cleaner
│       │   ├── blocking.py           # Country-partitioned multi-route inverted index blocker
│       │   ├── features.py           # 32-dimensional pairwise feature vector extraction
│       │   ├── model.py              # Gradient Boosted Trees model wrapper (XGBoost / LightGBM)
│       │   ├── train.py              # Hard negative mining and model training
│       │   ├── thresholding.py       # Macro F0.5 calibration & 1-to-1 bipartite exclusivity
│       │   ├── inference.py          # Disk-backed SQLite streaming test inference engine
│       │   ├── main.py               # CLI entry point for test pipeline execution
│       │   ├── data_loader.py        # Chunked, country-partitioned TSV reader
│       │   ├── evaluation.py         # Official competition Macro F0.5 metric evaluation
│       │   └── output.py             # Formatted TSV disk streaming
│       ├── README.md                 # Detailed module technical documentation
│       └── requirements.txt          # Pinned pipeline dependencies
├── models/
│   ├── final_entity_matcher.joblib   # Trained production GBDT model weights
│   └── model_metadata.json           # Calibrated decision thresholds and feature definitions
├── utils/
│   └── validate_submission.py         # Official submission integrity validator
├── Documentation_template.md          # Comprehensive formal challenge report
├── requirements.txt                   # Root dependency manifest
└── README.md                          # Master documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Karthick-564/Multilingual_Business_Entity_Resolution.git
cd Multilingual_Business_Entity_Resolution

# Setup clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Placement

Place your dataset under `dataset/` (or `student_resource/dataset/`):
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Run End-to-End Test Inference

Execute test candidate generation and matching inference:
```bash
python code/business_entity_resolution/src/main.py \
    --test-dir dataset/test \
    --model-path models/final_entity_matcher.joblib \
    --meta-path models/model_metadata.json \
    --output-dir output
```

This generates:
- `output/candidate_pairs.tsv`: Complete candidate sets for all test Source 1 entities.
- `output/matching_results.tsv`: Final high-precision entity resolution matches.

### 4. Submission Validation

Verify that generated files pass all competition integrity checks:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Expected Validation Result:**
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (141769 empty, 1590775 non-empty).
  candidate_pairs.tsv: 1732544 rows (42 empty, 1732502 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## ⚖️ Fair-Play & Model License Compliance

All models and components strictly adhere to competition guidelines:
- **Parameter Count:** Model contains ~63,000 parameters ($\ll 8\text{B}$ parameter limit).
- **Zero Closed-Source APIs:** No external web queries, commercial LLM APIs, or online databases.
- **Licenses:** All dependencies are permissively licensed under MIT, Apache-2.0, and BSD.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
