# CRISPR Guide RNA Design — Activity Prediction & Off-Target Specificity

A machine learning and algorithms project for multi-objective CRISPR guide RNA design.

The project is developed in stages:

1. **V1 — Application:** predict guide activity from sequence, evaluate RNA structure, estimate specificity, and rank guides using Pareto optimization.
2. **V2A — Evaluation improvement:** evaluate activity prediction using a leakage-free gene-level split, ensuring that test genes are completely unseen during training.
3. **V2B — Research extension:** develop and validate a more rigorous genome-scale off-target specificity engine using exact seed-indexed search and published CFD scoring. V2B is validated separately and is **not yet integrated into the Streamlit application**.

Efficiency and specificity are deliberately kept as **separate components**: a guide can cut well and still be unusable if it has close off-targets, so the objectives are evaluated independently rather than being collapsed into a single arbitrary score.

> **Status: V1/V2A application complete; V2B validated as a research extension.**

The current Streamlit application uses the V1 specificity backend. V2B development, validation, and benchmark reports are preserved under `V2B_RESEARCH/` as a research record.

---

## Results at a glance

### Activity prediction

| Model | Split | Spearman ρ |
|---|---|---:|
| Random Forest | random guide split (V1) | 0.728 |
| CNN | random guide split (V1) | 0.745 |
| Random Forest | **gene-level split (V2A)** | **0.718** |
| CNN | **gene-level split (V2A)** | **0.720** |

The V2A numbers matter more than the V1 numbers. A random guide split lets guides from the *same gene* appear in both train and test, which can cause information leakage.

The V2A evaluation instead uses a gene-level split, so the **4,034 test genes are entirely unseen during training**, corresponding to 10,944 test guides. The performance decrease was only approximately 0.01–0.025 Spearman ρ, supporting the interpretation that the models learned sequence-level determinants rather than simply memorizing gene-specific patterns.

Dataset: DeepHF, 54,956 guides across 20,168 genes, with gene labels joined from DeepHF Supplementary Data 1. 632 unmatched and 16 ambiguous guides were excluded.

### Specificity research extension (V2B)

| Metric | Result |
|---|---|
| Correctness | **197/197 tests**; exact vs brute force, **0 false negatives** |
| Memory | **12.3× reduction** — 370.6 → 30.03 bytes/site |
| Query speed | **4.0× faster** after optimisation (4.87 ms → 1.22 ms at 10 M sites) |
| vs brute force | **222× faster** on human chrM (3.89 → 0.0175 ms/guide) |
| Scale validated | **10 M sites**, 300 MB index, linear build, flat bytes/site |
| Scoring | Published Doench 2016 CFD matrix, bit-identical to the reference scorer |
| Application status | **Validated research extension, not yet integrated** |

The 10 M-site and hg38-scale results are validation/projection results from the separate V2B development workspace. **hg38 itself was not executed.**

---

## What V2B fixes from V1

The V1 specificity module had several limitations that motivated the V2B research extension:

| V1 limitation | V2B improvement |
|---|---|
| `specificity_proxy()` was a heuristic fallback rather than a genome search | Exact indexed genome search |
| Search was limited to chr22 | Per-chromosome indexing designed for genome-scale use |
| Used CFD-inspired position weights | Published Doench 2016 CFD matrix |
| Reverse-strand detection matched every C instead of true CCN | True NGG and reverse-strand CCN detection |

The reverse-strand issue was particularly consequential. During V2B validation, it over-called minus-strand sites by **2.92× on real chrM** (5,170 spurious vs 1,769 true) and **3.93× on synthetic sequence**.

These phantom sites could incorrectly penalize a guide's predicted specificity.

---

## Quick start

The current deployable application is the Streamlit V1/V2A prototype.

From the project directory:

```bash
streamlit run app.py