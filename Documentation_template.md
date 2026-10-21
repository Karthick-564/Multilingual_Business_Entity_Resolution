# ML Challenge 2026: Business Entity Resolution Solution Write-up

**Team Name:** Team QERNELS  
**Team Members:** Amazon ML Challenge Participant  
**Submission Date:** October 2026  

---

## 1. Executive Summary

We present a high-precision, multilingual, two-stage entity resolution pipeline engineered specifically for the Amazon ML Challenge 2026 under the competition **Macro $F_{0.5}$** evaluation metric. Our approach couples **script-agnostic phonetic & canonical normalization** (resolving 9 Indic scripts and accented French text) with **country-partitioned multi-attribute inverted index candidate blocking**, followed by a **precision-calibrated Gradient Boosted Decision Tree pairwise classifier** and **greedy bipartite 1-to-1 target exclusivity post-processing**. 

On held-out validation Source 1 entities, our pipeline achieves:
- **Validation Macro $F_{0.5}$ Score:** **`0.9761`**
- **Validation Precision:** **`0.9924` (99.24%)**
- **Validation Recall:** **`0.9454` (94.54%)**
- **Candidate Recall (Blocking Stage):** **`0.9648` (96.48%)**
- **Singleton Identification Accuracy:** **`0.9802` (98.02%)**

The entire pipeline operates in under 2.0 GB of host RAM by leveraging streaming disk caching and country-level partitioning, processing all 1,732,544 test Source 1 entities against ~10 million target records in ~90 minutes.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12.3 million training rows and 11.6 million test rows revealed critical structural characteristics of commercial business identity data:
1. **The Zero Cross-Country Law:** In an exhaustive audit of 741,074 ground truth links, **0.00% crossed country boundaries**. Entities only match within their exact country label (`US`, `India`, `France`). This allows strict country-level partitioning, eliminating 60%–70% of the Cartesian comparison space before string comparisons begin.
2. **Cross-Script Asymmetry:** 100% of Source 1 records are in Latin script. In contrast, in India records across Source 2 and Source 3, **over 18.9% of true matches are written in native Indic scripts** (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya). Without cross-script phonetic transliteration, standard lexical string overlap yields 0.00%, creating an immediate ~19% recall ceiling drop on Indian entities.
3. **Open-Set Country Behavior (France):** France appears only in the test set (~15.0% of test entities), presenting Latin text with accented characters, French legal abbreviations (`SARL`, `SAS`, `SCI`, `EURL`), and French address structures (5-digit postal codes, road terms like `Rue`, `Boulevard`, `Allée`, `bis`, `ter`).
4. **Universal Numeric & Primary House Number Anchors:** Regardless of language or script, digits (PIN codes, house/door numbers like `2621`, `RZ-142`, `1064`, `Unit 11`) are script-invariant and possess high discriminative power. Disagreement between primary house numbers on the same street is a decisive indicator of distinct co-located businesses or look-alikes.
5. **Macro $F_{0.5}$ Asymmetry & Singleton Risk:** In Macro $F_{0.5}$, Precision is weighted $2\times$ over Recall ($P$ is twice as important as $R$). Approximately 14.2% of Source 1 entities are singletons (0 matches). Correctly predicting an empty match list yields $1.0$, whereas even a single false-positive match drops the entity score to $0.0$. Conservative decision thresholds and target exclusivity are mandatory.

### 2.2 Solution Strategy
Our pipeline employs a decoupled, multi-stage architecture:
- **Approach Type:** Multi-Attribute Inverted Index Blocking + Supervised Gradient Boosted Decision Tree (XGBoost/LightGBM) + Global 1-to-1 Bipartite Exclusivity Matching.
- **Core Innovations:**
  1. *Script-Agnostic Phonetic Transliteration & Canonical Normalization:* Maps multi-script Indian tokens into standard Romanized phonetic representations, harmonizes legal corporate designations, and resolves street abbreviations (`St` $\leftrightarrow$ `Street`, `Rd` $\leftrightarrow$ `Road`, state abbreviations).
  2. *Composite Union Blocking with Frequency Pruning:* Eliminates $O(N \times M)$ pairwise comparison costs through 8 complementary inverted index keys, pruning high-frequency generic words (>150–300 count) to maintain ultra-compact candidate sets (~14.5 candidates per S1) while preserving a **96.48% candidate recall**.
  3. *Heterogeneous 32-Feature Representation:* Captures token sets, character 3-grams, Levenshtein, Jaro-Winkler, first-token conflicts (e.g. "Pizza Hut" vs "Dominos Pizza"), primary house number conflicts, and cross-source interactions.
  4. *Source-Calibrated Thresholding & Greedy 1-to-1 Target Exclusivity:* Enforces the real-world constraint that each target record (S2/S3) belongs to at most one Source 1 reference entity, assigning targets greedily to the highest-scoring candidate and eliminating duplicate attribution false positives.

---

## 3. Candidate Generation (Blocking)

To scale entity resolution without evaluating $1.7 \times 10^{13}$ Cartesian pairs, we partition candidates dynamically by country and retrieve candidates via 8 composite inverted-index routes:

- **Blocking Keys Used:**
  1. `compact_n`: Compact alphanumeric business name (all whitespace stripped) and sorted-token compact name (catches concatenated domain names and word reorderings, e.g. `moonboards.com` $\leftrightarrow$ `Moon Boards`).
  2. `core_n` & `n2`: Legal-suffix-stripped core business name and first two core name tokens.
  3. `n_tok`: Individual significant core name tokens ($\ge 4$ characters).
  4. `num_street`: Primary building/street number paired with rare street/locality words.
  5. `num_city`: Street number paired with city/state tokens.
  6. `name_num`: First name token paired with primary address number.
  7. `name_city`: First name token paired with city/state tokens.
  8. `addr_pair`: Unordered combinations of rare address lexical tokens.

### Blocking Performance
- **Average Candidates per S1 Entity:** 14.5 candidates (reduces Cartesian comparison space by **> 99.98%**).
- **Candidate Recall (on Validation Set):** **`96.48%`**.
- **How True Matches Were Preserved:**
  - Multi-attribute union ensures that if one attribute is corrupted (e.g., misspelled name), the candidate is rescued by another route (e.g., street number + postal locality).
  - High-frequency token pruning (>150–300 occurrences) filters out generic noise tokens (e.g., `India`, `Store`, `Enterprises`, `Road`, `Private`) without dropping distinctive entity keys.

---

## 4. Matching Model

### Features Used (32 heterogeneous pairwise features):
- **Name Features (14):** Exact clean string equality, exact core name equality, Levenshtein normalized distance, Jaro-Winkler similarity, RapidFuzz Token Sort ratio, Token Set ratio, Token Jaccard overlap, Min-overlap ratio, Character length difference, Length ratio, 4-character prefix equality, Character 3-gram Jaccard similarity, First-token match indicator ($\ge 0.80$), and First-token conflict penalty flag ($< 0.50$).
- **Address Features (10):** Address presence indicator, exact address equality, address Levenshtein similarity, address Jaro-Winkler similarity, Address Token Sort ratio, Address Token Set ratio, Address Token Jaccard overlap, shared numeric token count, numeric match indicator, and primary house/building number mismatch indicator.
- **Cross-Field & Source Interaction Features (8):** Exact core name with missing target address indicator, cross-field product ($\text{NameSim} \times \text{AddrSim}$), cross-field sum, cross-field max, cross-field min, Source 2 indicator, Source 3 indicator, and shared blocking key count.

### Model Architecture:
- **Primary Model:** Gradient Boosted Decision Trees (XGBoost / LightGBM) with histogram-based tree construction (`tree_method='hist'`), 350–500 estimators, max depth 7, 63 leaves, learning rate 0.035–0.04, feature subsampling 0.85, row subsampling 0.85.
- **Training Strategy:** Trained on 60,000 Source 1 entities (>800,000 training pairs) using ground-truth positive pairs and hard negatives mined directly from candidate blocking look-alikes.

### Threshold Selection & Exclusivity Deduplication:
- Decision thresholds calibrated directly on held-out validation data to maximize Macro $F_{0.5}$.
- Source-specific calibrated thresholds: $\tau_{S2} = 0.70 - 0.75$, $\tau_{S3} = 0.75 - 0.85$ (Source 3 exhibits higher noise, requiring higher thresholding).
- **Greedy 1-to-1 Bipartite Exclusivity:** Candidate pairs meeting threshold criteria are sorted globally by confidence descending. Each target record is assigned exclusively to the highest-scoring Source 1 entity, preventing spurious multi-attribution and driving precision to **> 99.1%**.

---

## 5. Results & Error Analysis

### Iterative Validation Progression:

| Experiment ID | Model / Architecture | Candidate Strategy | Features | Global Exclusivity | Precision | Recall | Macro $F_{0.5}$ | False Positives | False Negatives |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | LightGBM Baseline | SuperBlocking Top-15 | 28 features | No ($\tau=0.50$) | 0.9815 | 0.9521 | 0.968730 | 312 | 834 |
| **EXP-02** | LightGBM + Thresh Tuning | SuperBlocking Top-15 | 28 features | No ($\tau=0.82$) | 0.9922 | 0.9394 | 0.973518 | 129 | 1056 |
| **EXP-03** | LightGBM + Exclusivity | SuperBlocking Top-15 | 28 features | Yes ($\tau=0.82$) | 0.9932 | 0.9394 | 0.974255 | 112 | 1056 |
| **EXP-04** | LightGBM + Src Thresholds | SuperBlocking Top-15 | 28 features | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.80$) | 0.9920 | 0.9425 | 0.974534 | 133 | 1002 |
| **EXP-05** | Ablation: Name Only | SuperBlocking Top-15 | 14 Name feats | Yes | 0.8859 | 0.8734 | 0.889718 | 1959 | 2206 |
| **EXP-06** | Ablation: Address Only | SuperBlocking Top-15 | 10 Addr feats | Yes | 0.9690 | 0.8494 | 0.919123 | 474 | 2624 |
| **EXP-07** | HistGradientBoosting | SuperBlocking Top-15 | 28 features | Yes | 0.9938 | 0.9391 | 0.974371 | 102 | 1061 |
| **EXP-08** | Deep LightGBM | SuperBlocking Top-15 | 28 features | Yes | 0.9914 | 0.9440 | 0.974164 | 143 | 976 |
| **EXP-09** | Enhanced Feats + Top-20 | SuperBlocking Top-20 | 32 features | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.85$) | **0.9924** | **0.9454** | **0.976105** | **126** | **951** |
| **EXP-10** | Ensemble (LGB + XGB) | SuperBlocking Top-20 | 32 features | Yes | **0.9932** | **0.9439** | **0.976653** | **113** | **977** |

- **Best Production Macro $F_{0.5}$:** **`0.9761`** (Single Model) / **`0.9766`** (Ensemble)
- **Singleton Accuracy:** **`98.02%`** (correctly predicting empty matches for singletons, preserving full $1.0$ entity scores).
- **Error Analysis & Edge Cases:**
  - *Mitigated False Positives:* Co-located businesses sharing generic brand or category names (e.g., adjacent suites in shopping plazas or business parks) were resolved by our primary street number mismatch feature and first-token conflict detector.
  - *Remaining False Negatives:* Highly abbreviated acronyms where both the trading name and the physical address were severely corrupted or truncated in Source 2/3.

---

## 6. Conclusion

By integrating script-agnostic phonetic normalization, country-partitioned multi-attribute inverted index blocking, 32 precision-engineered features, and 1-to-1 bipartite target exclusivity, **Team QERNELS** achieves superior entity resolution performance exceeding **0.976 Macro $F_{0.5}$** with **>99.2% precision**. The architecture is fully streaming, deterministic, lightweight (<2 GB RAM), and 100% compliant with competition fair-play rules.
