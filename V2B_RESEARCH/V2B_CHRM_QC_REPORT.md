# V2B Specificity — Real-Reference QC Report (Option A: human chrM)

**Verdict: PASS** — 5/5 groups, 28 chrM test cases, 0 failures.
Combined suite (synthetic + chrM): **119/119 pass**.

| Field | Value |
|---|---|
| Reference | **NC_012920.1** (rCRS), human mitochondrion, complete genome |
| Length | 16,569 bp — sequence md5 `c68f52674c9fb33aef52dcf399755519` |
| FASTA on disk | 16,864 bytes (16.5 KB) |
| hg38 | **Not downloaded.** chrM is the only real sequence in the workspace. |
| On-disk index | **None.** All indexes built in memory and discarded. |
| Workspace total | **493 KB** (0.38 % of the 128 MB limit) |
| V1 / V2A / app.py | **Untouched** — nothing exists outside `v2b_specificity/` |
| Reproduce | `cd v2b_specificity && python3 run_chrM_validation.py` |

---

## 1. Results by requirement

| # | Requirement | Group | Tests | Result |
|---|---|---|---|---|
| 1–3 | Mito reference only, no hg38, no large index | A1 | 4 | **PASS** |
| 4 | True NGG and CCN detection on real sequence | A2 | 6 | **PASS** |
| 5 | N-containing regions and repeats | A3 | 7 | **PASS** |
| 6 | Seed-index vs brute-force equivalence | A4 | 5 | **PASS** |
| 7 | CFD scoring on real genomic candidate sites | A5 | 6 | **PASS** |
| 8 | Memory and runtime benchmarks | — | measured | **PASS** |
| 9 | Small QC report | — | this file | **PASS** |

---

## 2. QC metrics on real chrM

| Metric | Value |
|---|---|
| Length / GC% | 16,569 bp / 44.36 % |
| Base counts | A 5,124 · C 5,181 · G 2,169 · T 4,094 · **N 1** |
| N position | **1-based 3107** (the rCRS placeholder) |
| **Total sites** | **2,193** (132.36 per kb) |
| Forward (NGG) | 424 |
| Reverse (CCN) | 1,769 |
| C/G ratio (light strand) | 2.389 |
| Sites suppressed by N filter | 1 |
| Duplicate protospacers | **0** (max copies = 1) |
| Homopolymer runs ≥5 bp | 114 (longest: A×8 at 1-based 12418) |
| Guide GC | mean 0.449, range 0.00–0.80 |
| **Min pairwise Hamming** | **4** |
| True off-target pairs ≤3 mm | **0** |

---

## 3. Real-data findings

### 3.1 Strand asymmetry is genuine biology, not a regression

424 forward vs 1,769 reverse sites looks alarming — a 4.2× skew that on synthetic
uniform DNA would suggest the old bug had returned. It is real. The mtDNA light
strand (the strand deposited in GenBank) is C-rich: **C/G = 2.39**. Since forward
sites need `GG` and reverse sites need `CC`, the site counts must follow the
dinucleotide counts, and they do exactly:

| | dinucleotide count | sites found | difference |
|---|---|---|---|
| `GG` | 425 | 424 | 1 (no room for a full upstream 20-mer) |
| `CC` | 1,771 | 1,769 | 2 (1 edge, 1 N-filtered) |

This is asserted as a test, so the ratio is now pinned to its biological cause
rather than left as an unexplained number. **This is exactly the kind of signal
the synthetic stage could not produce** — uniform random sequence has C/G ≈ 1.

### 3.2 The V1 bug is still catastrophic on real sequence

Running the preserved V1 oracle on rCRS: **5,170 reverse "sites" vs 1,769 real
ones — 2.92× over-calling**, and every spurious entry has a PAM that does not
read NGG. On a C-rich real genome the old code would have invented ~3,400
phantom off-target loci in a 16.6 kb sequence alone.

### 3.3 The N filter is provably exact

The single N at 3107 sits inside the window `TTCTATCTAC N TTCAAATTC`. Exactly one
CCN 23-mer covers it. The scanner suppresses exactly that one site, and patching
`N→A` recovers exactly one site (2,193 → 2,194). No site in the output overlaps
position 3107 in either protospacer or PAM.

### 3.4 chrM has no internal off-targets — and that is a real result

The minimum pairwise Hamming distance between any two of the 2,193 protospacers
is **4**. Distance distribution: 6 pairs at 4 mm, 45 at 5, 240 at 6, 1,141 at 7,
4,414 at 8. Consequently **every real chrM guide is uniquely specific within
chrM** at a ≤3-mismatch budget: each finds itself exactly once and scores
specificity 100.0.

Two consequences worth recording:

* It confirms the searcher produces no false positives on real sequence — a
  naive or buggy matcher would manufacture hits here.
* It means chrM alone **cannot** validate multi-hit ranking. That case was
  covered in the synthetic stage (planted 0/1/2/3-mm loci) and was additionally
  re-exercised here by mutating real chrM guides by 1–3 nt and confirming
  index/brute-force agreement on the resulting non-trivial hit sets.

### 3.5 Repeats and low complexity

114 homopolymer runs ≥5 bp (longest A×8 at 12418) and the classic `(CA)₅` tract
at 1-based 514 are all traversed without error; overlapping sites remain
internally consistent and re-derivable from coordinates. Guide GC spans 0.00–0.80,
including a 0 % GC guide from a poly-A tract — no crashes, no degenerate scores.

---

## 4. Benchmarks (requirement 8)

Timings on the sandbox CPU, Python 3.13. Memory via `tracemalloc` (Python
allocations only).

### Build phase
| Metric | Value |
|---|---|
| Site scan (whole chrM, both strands) | **0.070 s** |
| Site scan peak memory | 779 KB |
| Index build (2,193 sites, 4 blocks) | **0.030 s** |
| Index peak memory | 610 KB |
| **Index cost per site** | **285 bytes** |
| Keys per block | 782 / 809 / 790 / 792 |

### Query phase (300 real guides, max_mm = 3)
| Metric | Brute force | Seed index |
|---|---|---|
| Total | 1.1667 s | **0.0052 s** |
| Per guide | 3.889 ms | **0.0175 ms** |
| Throughput | 257 guides/s | **57,176 guides/s** |
| **Speedup** | — | **222×** |

Candidate pruning: mean 13.6 candidates per query (**0.62 %** of sites), max 30.

### Scoring
| Metric | Value |
|---|---|
| CFD pairs scored | 10,000 real-site pairs |
| Per pair | 9.19 µs (**108,817 pairs/s**) |
| **All 2,193 chrM guides, full specificity profile** | **0.059 s** |

### Extrapolation note (not a plan to run it)
285 bytes/site is a pure-Python dict-of-lists figure. At the projected ~388 M
hg38 sites that structure implies ~110 GB — confirming with real measurement
what the synthetic report predicted analytically. The packed/offset-array design
in `V2B_SYNTHETIC_VALIDATION.md` §4 remains mandatory before any genome-scale
build. **No hg38 work was started, per instruction.**

---

## 5. Files added in this stage (all tiny)

```
data/reference/chrM_rCRS.fa            16.5 KB   NC_012920.1 rCRS
src/v2b/fasta.py                        1.8 KB   tiny-FASTA reader, 5 MB hard guard
tests/test_chrM_real_reference.py      14 KB     28 chrM test cases
run_chrM_validation.py                 11 KB     validation + benchmark runner
reports/chrM_validation_report.json     3 KB     machine-readable results
reports/V2B_CHRM_QC_REPORT.md          this file
```

`fasta.py` refuses any FASTA over 5 MB, so a genome file cannot be silently
ingested into this workspace later.

---

## 6. Status and stopping point

* Requirements 1–9: **all satisfied, all PASS.**
* V1 unchanged. V2A unchanged. V2B **not** integrated into `app.py`.
* **Stopped here. hg38 not started**, as instructed.

The algorithm is now validated on both synthetic ground truth (where every
answer is known by construction) and real human sequence (with genuine strand
skew, an ambiguity code, homopolymers and low-complexity tracts). Remaining work
before genome scale is engineering — the packed index — not correctness.
