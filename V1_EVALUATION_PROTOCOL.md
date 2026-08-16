# V1 EVALUATION PROTOCOL (locked, for fair V1 vs V2A comparison)

**Dataset:** DeepHF (Wang et al. 2019), WT-SpCas9 efficiency.
- Raw file: `41467_2019_12281_MOESM3_ESM.xlsx` (Supplementary Data 2)
- Cleaned file: `data/deepHF_clean.csv` — columns `Sequence, Activity`
- Cleaning: keep rows with non-null `gRNA_Seq` + `Wt_Efficiency`; sequence length >= 19;
  drop exact-duplicate `gRNA_Seq` (keep first)
- N = **55,604 unique WT guides**

**Features (preprocessing):**
- One-hot encoding of the 20-nt guide → 4 × 20 = 80 binary features (A/C/G/T; U→T)
- + GC content (fraction of G+C over the 20 nt) → 1 feature
- Total feature vector: 81

**Split (V1 — guide-level, random):**
- `train_test_split(..., test_size=0.2, random_state=42)` on GUIDES (not genes)
- Train 80% → then `train_test_split` of that 80% into train/val at 0.125 → 80/10/10 overall
- Test set NEVER touched during training or hyperparameter selection
- CNN best epoch selected on VALIDATION set only (not test)

**Models (V1):**
- RandomForestRegressor: n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
- CNN (1D): Conv1d(4→32, k=3, pad=1) → ReLU → Conv1d(32→64, k=3, pad=1) → ReLU →
  AdaptiveAvgPool1d(1) → Flatten → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid
- Optimizer Adam lr=1e-3; loss MSE; batch 256 (V1 used batch 64 for 12 epochs;
  later evaluate.py used batch 256, 15 epochs, val-best epoch restore)
- **NOTE:** V1's published numbers (RF 0.728 / CNN 0.745) came from `evaluate.py`
  (batch 256, 15 epochs, val-based epoch selection). This is the protocol we lock for V2A.

**Metrics:**
- Spearman rank correlation (scipy.stats.spearmanr) on the held-out TEST set
- RMSE on the held-out TEST set
- Both reported for RF and CNN

**V1 results (from `results/eval_metrics.csv`):**
- DeepHF random 80/10/10: RF Spearman **0.728**, RMSE 0.146; CNN Spearman **0.745**, RMSE 0.139
- (Doench: random 0.613/0.298; gene-level 0.453/0.280 — used for the leakage narrative, not the V1-vs-V2A baseline)

**What is NOT part of V1's evaluation protocol (and must NOT be changed for V2A):**
- No gene-level split (that's the V2A change)
- No data augmentation, no feature selection, no extra hand-crafted features beyond one-hot+GC
- No off-target/specificity features in the efficiency model
- No weighted multi-objective scoring in the model evaluation (that's the app layer)

---

# V2A PROTOCOL (planned — only the split changes)

**Dataset:** same DeepHF 55,604; mapped to genes via design table:
- `results/v2/gene_labels.csv` — `Sequence, PAM, Activity, Gene_Symbol, Entrez_ID`
- N = **54,956** unambiguous single-gene guides (after excluding 632 unmatched + 16 ambiguous)
- 20,168 genes

**Split (V2A — gene-level, NOT random):**
- Split GENES: 80% of genes → train, 20% → test (seeded, random_state=42)
- ALL guides belonging to a test gene go to test; NO guide from a test gene in train
- Train genes → 80/20 split of GUIDES into train/val (val only for CNN epoch selection)
- Test set untouched

**Models / features / metrics:** IDENTICAL to V1 (one-hot + GC; RF 200/12; same CNN;
Spearman + RMSE on test; val-based CNN epoch selection)

**Expected effect (to interpret, not to pre-judge):**
- Gene-level split removes same-gene guides from training; if the model relied on
  gene-specific sequence patterns, Spearman will drop relative to V1 (0.728/0.745).
- The size of the drop is the honest measure of cross-gene generalization.
- We will ALSO report a secondary analysis on the 386 genes with >= 5 guides.
  NOTE (verified against the code): `v2_train_eval.py` computes ONE aggregate
  Spearman over all pooled test guides of that subset — it does NOT compute
  per-gene Spearman values and aggregate them. The wording reflects this.
