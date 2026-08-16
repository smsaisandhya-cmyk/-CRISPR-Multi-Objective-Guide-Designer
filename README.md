# CRISPR Guide RNA Design — Activity Prediction & Off-Target Specificity

A two-part machine learning and algorithms project for CRISPR guide RNA design:

1. **Activity prediction** (V1 / V2A) — predict on-target cutting efficiency from
   guide sequence, evaluated under a leakage-free gene-level split.
2. **Specificity engine** (V2B, *this repository*) — an exact sequence-based
   off-target search with published CFD scoring, designed for genome-scale
   indexing and validated on real human chrM and synthetic datasets up to
   10 M sites.

Efficiency and specificity are deliberately kept as **separate components**: a
guide can cut well and still be unusable if it has close off-targets, so the two
scores are reported independently rather than blended.

> **Status: complete and frozen.** All figures below are measured, reproducible
> outputs of the scripts in this repo. Nothing is projected except where a line
> explicitly says "projection".

---

## Results at a glance

### Activity prediction

| Model | Split | Spearman ρ |
|---|---|---|
| Random Forest | random guide split (V1) | 0.728 |
| CNN | random guide split (V1) | 0.745 |
| Random Forest | **gene-level split (V2A)** | **0.718** |
| CNN | **gene-level split (V2A)** | **0.720** |

The V2A numbers matter more than the V1 numbers. A random guide split lets guides
from the *same gene* appear in both train and test, which leaks. Re-splitting so
that **4,034 test genes are entirely unseen** (10,944 test guides) cost only
~0.01–0.025 ρ — evidence the models learned sequence determinants rather than
memorising genes.

Dataset: DeepHF, 54,956 guides across 20,168 genes, gene labels joined from DeepHF
Supplementary Data 1 (632 unmatched + 16 ambiguous guides excluded).

### Specificity engine (V2B)

| Metric | Result |
|---|---|
| Correctness | **197/197 tests**; exact vs brute force, **0 false negatives** |
| Memory | **12.3× reduction** — 370.6 → 30.03 bytes/site |
| Query speed | **4.0× faster** after optimisation (4.87 ms → 1.22 ms at 10 M sites) |
| vs brute force | **222× faster** on human chrM (3.89 → 0.0175 ms/guide) |
| Scale validated | **10 M sites**, 300 MB index, linear build, flat bytes/site — hg38 was **projected, not executed** |
| Scoring | Published Doench 2016 CFD matrix, bit-identical to the reference scorer |

---

## What this fixes from V1

The V1 specificity module had three defects that made its output unusable as a
real specificity estimate:

| V1 defect | V2B |
|---|---|
| `specificity_proxy()` was a **heuristic fallback**, not a genome search | Real indexed genome search |
| Searched **only chr22** | Per-chromosome shards, designed for genome-scale indexing |
| **CFD-inspired position weights**, not the published matrix | Published Doench 2016 CFD tables |
| Reverse-strand bug: matched **every C** instead of true CCN | True CCN detection |

The reverse-strand bug was the most consequential: it over-called minus-strand
sites by **2.92×** on real chrM (5,170 spurious vs 1,769 true) and **3.93×** on
synthetic sequence. Every one of those phantom sites would have counted against a
guide's specificity score.

---

## Quick start

```bash
cd v2b_specificity
python3 -m pytest                 # 197 tests
python3 run_validation.py         # synthetic validation + measurements
python3 run_chrM_validation.py    # real reference (human chrM)
```

```python
import sys; sys.path.insert(0, "src")
from v2b import SeedIndex, guide_specificity

idx = SeedIndex.from_sequence(sequence, chrom="chr1", max_mm=3)
rep = guide_specificity("GACGTTGCAGTCACTAGGCA", idx, max_mm=3)
rep["by_mismatch"]        # {0: .., 1: .., 2: .., 3: ..}
rep["specificity_score"]  # 100 / (1 + sum of off-target CFD)
```

Scale-oriented path (packed + memory-mapped):

```python
from v2b.packed import PackedIndex
from v2b.store import save_index, open_index

idx = PackedIndex(max_mm=3)
idx.add_shard("chr1", sites)        # one contig at a time; peak RAM stays flat
save_index(idx, "idx/")
with open_index("idx/") as mi:      # ~0.3 ms load regardless of size
    hits = mi.search(guide, max_mm=3)
```

---

## How it works

Off-target search is **exact**, not heuristic, via the pigeonhole principle: a
20-nt guide split into `max_mm + 1 = 4` blocks of 5 nt cannot accumulate 3
mismatches without leaving **at least one block exact**. Indexing all four blocks
and unioning their hits therefore cannot miss a true off-target — verified
directly against brute force at every stage.

Three engineering layers sit on top:

- **2-bit packing** — a 20-mer in one `uint64`; mismatches by XOR + masked popcount.
- **CSR postings** — counting-sorted offset/id arrays instead of dict-of-lists.
- **mmap shards** — `np.frombuffer` views over a memory map; no copy, no pickle.

Full design rationale, format spec, and benchmark methodology:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---
## Components

| Stage | What it is | Status |
|---|---|---|
| V1 | CNN activity prediction (DeepHF 55,604), chr22 CFD-inspired specificity, ViennaRNA structure, and Pareto ranking | ✅ **Live in the Streamlit app** |
| V2A | Gene-level split evaluation (RF 0.718 / CNN 0.720, 4,034 held-out genes) | ✅ **Complete** |
| V2B | Seed-index + published-CFD off-target specificity research prototype | ⏳ **Validated research extension, not yet integrated into the app** |

**Note:** The deployed prototype currently uses the V1 specificity backend. V2A is an evaluation/validation stage rather than a separate app component. V2B is the next specificity upgrade and has been validated experimentally at the prototype level, but full genome-scale integration into the deployed application remains future work.

## Validation history

Each stage was gated on the previous one passing.

| Stage | What was proven | Result |
|---|---|---|
| 1. Synthetic | PAM detection, pigeonhole index, CFD correctness | PASS, 88 tests |
| 2. Real reference (chrM) | 2,193 real sites, the N at 3107, repeats | PASS 5/5, 222× |
| 3. Packed index | Identical results at 12.3× less memory | PASS 5/5, 162 tests |
| 4. mmap store | Bit-identical round-trip, O(1) load | PASS 10/10, 197 tests |
| 5. Scale stress | 1/5/10 M sites; 30.03 B/site flat | PASS |
| 6. Filter analysis | Theory + benchmark of 5 candidate strategies | Analysis |
| 7. Optimisation | 4.0× query speedup, zero format change | PASS |

Per-stage reports live in [`reports/`](reports/), each with a machine-readable
`.json` beside the markdown.

**Stage 5 found the real bottleneck.** Storage scales fine; *query time* is linear
in index size because candidates examined follow `4n/1024` (matched to within 1%).
Stage 6 compared five ways to cut that, and Stage 7 implemented the two that
required no format change — a sort+diff dedupe replacing `np.unique` (which spends
89% of its time in a hash path before sorting anyway) and a threshold-guarded
one-pass CFD kernel.

### Known limits

- Query latency is still **linear in index size**. Stage 7 cut the constant ~4×,
  not the slope. hg38 projects to ~50 ms/query (from ~198 ms), plus a ~143 µs/query
  fixed floor across 24 shards that no filter can remove.
- Going sub-linear needs a two-block-match filter (Stage 6 strategy S2, measured
  16.7× end-to-end) — deliberately **not adopted**, because it costs ~16.5 GB extra
  at hg38 scale and changes the on-disk format.
- **hg38 has never been run.** All scale figures use synthetic random sequence,
  which lacks the repeat structure that will skew real candidate counts upward.
- Indels are not modelled; substitutions only.

---

## Repository layout

```
src/v2b/
  pam.py       NGG / true CCN site detection
  seq.py       encoding, reverse complement
  index.py     reference SeedIndex (dict-based, the correctness oracle)
  search.py    search + specificity report
  cfd.py       published Doench 2016 CFD tables
  packed.py    columnar packed index (2-bit, CSR, popcount) — the fast path
  store.py     .v2bs on-disk format + mmap loader
  fasta.py     FASTA reader (rejects >5 MB by design)
tests/         197 tests
reports/       per-stage reports (.md + .json)
backups/       pre-optimisation snapshot + checksums
```

## Reproducibility

Fixed seeds throughout; every reported number is regenerated by a script in this
repo. Python 3.13, numpy. No network access needed except the one-time chrM fetch
(already vendored at `data/reference/chrM_rCRS.fa`, 16.5 KB).

Deliberate guardrails: `fasta.py` refuses FASTA >5 MB, no genome or serialized
index is committed, and all large-scale tests generate sequence in memory from
seeds. Total repo footprint ~1 MB.
