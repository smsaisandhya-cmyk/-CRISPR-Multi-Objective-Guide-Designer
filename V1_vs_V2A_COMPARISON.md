# V1 vs V2A Comparison — DeepHF WT-SpCas9 Guide Activity Prediction

**Date:** 2026-08-16
**Project:** CRISPR multi-objective guide design (laptop-only, for A*STAR application)
**Question:** Does the model generalize to guides from *completely unseen genes*?

---

## 1. The single-variable change

| | V1 | V2A |
|---|---|---|
| Split unit | **guides** (random) | **genes** |
| Split | 80/10/10 of 55,604 guides | 80% of genes → train, 20% → test; all guides of a test gene held out; train genes split 80/20 into train/val |
| Test set | 5,561 guides (random) | 10,944 guides from **4,034 unseen genes** |
| Features | one-hot(4×20) + GC | **identical** |
| RF | 200 trees, max_depth 12 | **identical** |
| CNN | 2×Conv1d(4→32→64)+AvgPool+FC, Adam 1e-3, MSE, batch 256, 15 epochs, val-best restore | **identical** |
| Metrics | Spearman + RMSE on held-out test | **identical** |

The ONLY difference is the data-splitting strategy.

---

## 2. Headline result — V2A_all (all 54,956 unambiguous single-gene guides; 20,168 genes)

| Model | V1 (random split) | V2A (gene split) | Δ (V2A − V1) | Interpretation |
|---|---|---|---|---|
| RandomForest | 0.728 | **0.718** | **−0.010** | Negligible drop — RF rules are largely gene-agnostic |
| CNN | 0.745 | **0.720** | **−0.025** | Small drop — CNN also generalizes well across unseen genes |

- RMSE: V1 RF 0.146 / CNN 0.139 → V2A RF 0.149 / CNN 0.146 (essentially unchanged).
- Test set: 10,944 guides from **4,034 genes never seen in training** (vs 5,561 random guides in V1).
- **Conclusion (cautious):** The small drop (~0.01–0.025) when testing on completely unseen genes **supports good cross-gene generalization** and **suggests** that the learned sequence–activity relationship generalizes well to unseen genes (e.g., position-dependent nucleotide preference, GC, motif effects), rather than being driven by gene identity. This is an interpretation supported by the result, not a proof — we cannot rule out residual gene-correlated sequence features without further analysis.

---

## 3. Secondary analysis — V2A_genes5plus (genes with ≥5 guides; 386 genes, 1,971 guides)

| Model | V1 (random) | V2A (gene split) | Δ | Note |
|---|---|---|---|---|
| RandomForest | 0.728 | **0.594** | −0.134 | Larger drop — but on a small subset (394 test guides, 78 genes) |
| CNN | 0.745 | **0.367** | −0.378 | Large drop — **data-hunger effect**, not split alone |

**Interpret with care (do NOT read as "gene split is worse at scale"):**
- This subset has only 1,971 guides; CNN trained on just 1,261 guides → the same "deep learning needs data" effect seen on Doench (CNN 0.36 at 1,841 guides).
- The larger RF drop (−0.134) here vs the full set (−0.010) suggests that within well-covered genes there is *some* gene-specific signal, but it is confounded with small-sample noise.
- **This is a secondary sanity check, not the headline.** The full-set result (V2A_all) is the authoritative comparison.

---

## 4. What this means scientifically

1. **Cross-gene generalization appears strong.** The small performance drop **supports** good cross-gene generalization — the learned sequence–activity relationship appears to transfer to unseen genes — though this is a suggestive finding, not a proof of the absence of gene-specific memorization.
2. **V1's random-split numbers were not meaningfully inflated by gene leakage.** The gap (0.01–0.025) is small, suggesting the random-split estimates (V1) were close to what a stricter gene-level split (V2A) yields.
3. **Data-hunger of DL is reproducible:** CNN ≈ RF on the full set (0.720 vs 0.718), CNN collapses on small subsets (0.367) — consistent with our Doench (1,841) results.
4. **Both models remain usable at scale;** the choice of RF vs CNN is not the main axis of generalization — the split strategy is.

---

## 5. Reproducibility record (for the report)

- Dataset: DeepHF WT-SpCas9 efficiency, 55,604 unique guides; mapped to genes via the **original design table** (Supplementary Data 1, MOESM2) — `results/v2/gene_labels.csv` (54,956 unambiguous single-gene guides, 20,168 genes).
- Mapping exclusions (documented): 632 unmatched (1.14%), 16 ambiguous (0.03%), 85 malformed design rows.
- V2A gene split: `16134 train genes / 4034 test genes` (80/20, seeded).
- No gene leakage (asserted in script; verified "No gene leakage between train/test ✅").
- Scripts: `src/v2_map_genes.py` (mapping), `src/v2_train_eval.py` (gene-level training/eval).
- V1 protocol locked in `V1_EVALUATION_PROTOCOL.md`.
- Metrics files: `results/eval_metrics.csv` (V1), `results/v2/v2a_metrics.csv` (V2A).
