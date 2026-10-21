# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

**Team Name:** Team QERNELS  
**Track:** Business Entity Resolution  
**Status:** Validated & Production-Ready (`validate_submission.py` PASS)

---

## 1. Overview & Architecture

This repository contains the complete, self-contained end-to-end pipeline developed by **Team QERNELS** for the **Amazon ML Challenge 2026 Business Entity Resolution** task.

The system is optimized directly for the competition metric, **Macro $F_{0.5}$**, where **Precision is weighted $2\times$ over Recall**, and singleton entities (entities with 0 ground-truth matches) require conservative abstention to avoid catastrophic false-merge penalties.

### Pipeline Stages:
1. **Script-Agnostic Phonetic Normalization (`normalization.py`):** Unifies multilingual Indian scripts (Devanagari, Telugu, Tamil, Kannada, Gujarati, Bengali, Malayalam, Punjabi, Oriya) into Romanized phonetic ASCII via unidecoding, harmonizes French diacritics and address tokens, and strips international legal suffixes (`Pvt Ltd`, `LLC`, `SARL`, `SAS`, `GmbH`).
2. **Country-Partitioned Multi-Attribute Inverted Indexing (`blocking.py`):** Partitions candidate generation strictly by country (exploiting the 0.00% cross-country matching law), using 8 composite indexing routes (compact names, root n-grams, street number + street token, name token + postal/locality) with frequency capping (>150–300 count pruning). Achieves **>96.4% candidate recall** while reducing the Cartesian comparison space by **99.98%**.
3. **Supervised Pairwise Classifier (`features.py`, `model.py`):** Extracts 32 heterogeneous features spanning exact core matches, character 3-gram similarities, Levenshtein, Jaro-Winkler, primary building/street number match/conflict vetoes, first-token conflicts, and cross-source interactions. Predicted using an ensemble of Gradient Boosted Decision Trees (XGBoost / LightGBM) trained on hard negatives mined directly from blocking look-alikes.
4. **Calibrated Thresholding & Bipartite 1-to-1 Target Exclusivity (`thresholding.py`):** Calibrates separate decision thresholds for Source 2 ($\tau_{S2} = 0.70$) and Source 3 ($\tau_{S3} = 0.75$), followed by greedy highest-confidence 1-to-1 target assignment to prevent spurious multiple-attribution, achieving **>99.1% validation precision**.

---

## 2. Fair-Play & Model License Compliance

All algorithms and models used in this pipeline strictly adhere to the competition rules, are **≤ 8B parameters individually**, and are released under permissive open-source licenses (MIT / Apache-2.0 / ISC):

| Component / Model | Type | Parameter Count | License | Permitted |
| :--- | :--- | :---: | :---: | :---: |
| **Inverted Index Multi-Route Blocker** | Algorithmic Indexing | 0 (Heuristic / Hashing) | Custom / Apache-2.0 | Yes |
| **XGBoost Classifier** | Gradient Boosted Trees | ~63,000 parameters (< 0.0001B) | Apache-2.0 | Yes (≤ 8B) |
| **LightGBM Classifier** | Gradient Boosted Trees | ~35,000 parameters (< 0.0001B) | MIT | Yes (≤ 8B) |
| **RapidFuzz Engine** | C++ String Metrics | 0 (Algorithmic) | MIT | Yes |
| **Text-Unidecode / AnyAscii** | Phonetic Transliteration | 0 (Deterministic Table) | Artistic / MIT / ISC | Yes |
| **Bipartite Exclusivity Matcher** | Greedy Graph Matching | 0 (Algorithmic) | Apache-2.0 | Yes |

*No closed-source APIs, proprietary LLMs, or pre-trained models exceeding 8B parameters were used.*

---

## 3. Environment Setup & Requirements

### Hardware Requirements
- **RAM:** Minimum 8 GB (Pipeline uses streaming & SQLite caching, peak RAM usage is **< 2.0 GB**).
- **Disk:** Minimum 5 GB free disk space for temporary candidate caching and output TSV files.
- **Compute:** Multi-core CPU (supports 4 to 20+ threads). Optional NVIDIA CUDA GPU for accelerated tree scoring.
- **Expected End-to-End Runtime:**
  - Training: ~15 – 25 minutes
  - Test Inference (all 1.73M S1 entities across 11.5M records): ~85 – 100 minutes

### Environment Installation
Create a clean Python 3.10+ virtual environment and install the exact pinned dependencies:

```bash
# 1. Create and activate clean virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate

# 2. Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction Instructions

To reproduce both `output/candidate_pairs.tsv` and `output/matching_results.tsv` from the raw dataset, execute the following commands in order:

### Step 1: Model Training (Hard Negative Mining & Calibration)
To train the gradient-boosted pairwise model from scratch using the raw training dataset:

```bash
python src/train.py \
    --train-dir dataset/train \
    --n-train 60000 \
    --n-val 10000 \
    --model-type xgboost_gpu \
    --model-out models/final_entity_matcher.joblib \
    --meta-out models/model_metadata.json
```
*(If running on a CPU-only environment, omit `_gpu` to use `--model-type xgboost` or `--model-type lightgbm`)*.

### Step 2: Test Dataset Inference & Output Generation
Run the production inference pipeline on the test dataset. This performs country-partitioned blocking, feature extraction, GPU/CPU batch scoring, and bipartite target exclusivity post-processing:

```bash
python src/main.py \
    --test-dir dataset/test \
    --model-path models/final_entity_matcher.joblib \
    --meta-path models/model_metadata.json \
    --output-dir output \
    --batch-size 10000 \
    --top-k 20
```

This generates:
- `output/candidate_pairs.tsv`: Full candidate sets for all 1,732,544 test Source 1 entities.
- `output/matching_results.tsv`: Final high-precision entity resolution matches.

### Step 3: Run the Official Submission Validator
Verify the generated output files against every competition integrity and formatting rule:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
**Expected Output:**
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (141769 empty, 1590775 non-empty).
  candidate_pairs.tsv: 1732544 rows (42 empty, 1732502 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## 5. File & Directory Layout

```
Team_QERNELS_submission/
├── output/
│   ├── matching_results.tsv       # Final matches (byte-identical to leaderboard submission)
│   └── candidate_pairs.tsv        # Complete blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py        # Package initialization
│       │   ├── normalization.py   # Script-agnostic phonetic & text normalization
│       │   ├── blocking.py        # Multi-attribute inverted index candidate generation
│       │   ├── features.py        # 32 heterogeneous pairwise feature extraction
│       │   ├── model.py           # Classifier architectures (XGBoost / LightGBM)
│       │   ├── train.py           # Hard negative mining and model training
│       │   ├── thresholding.py    # Macro F0.5 calibration & 1-to-1 bipartite exclusivity
│       │   ├── inference.py       # Batched streaming inference engine with disk caching
│       │   ├── main.py            # CLI entry point for test pipeline execution
│       │   ├── output.py          # TSV formatting and disk streaming
│       │   └── evaluation.py      # Macro F0.5 evaluation implementation
│       ├── README.md              # Reproduction instructions & license audit (this file)
│       └── requirements.txt       # Exact pinned dependencies
└── Documentation_template.md      # Completed methodology write-up
```
