# V2B Stage 3 — Memory-Efficient Packed Index

**Result: PASS** — 5/5 validation groups, 43 new tests, 162/162 total suite.
The packed index reproduces the validated `SeedIndex` + `seed_index_search`
**exactly**, at **12.3× lower memory**.

No hg38 was downloaded or indexed. V1, V2A and `app.py` untouched.
Machine-readable results: `reports/packed_index_report.json`.

---

## 1. What was built

`src/v2b/packed.py` — a columnar, numpy-backed replacement for the Python
dict-of-lists index. Nothing in `index.py` or `search.py` was modified; the old
implementation remains the reference oracle.

**2-bit packing.** `A=0, C=1, G=2, T=3`, position 1 in the most significant
bits, so a 20-nt protospacer is 40 bits in a single `uint64`. Sites are stored
as four parallel columns and `Site` dataclasses are *reconstructed on demand*:

| column | dtype | bytes/site |
|---|---|---|
| `words` (packed 20-mer) | `uint64` | 8 |
| `strand` | `uint8` | 1 |
| `pam_start` | `int32` | 4 |
| `pam_n` (the N of NGG) | `uint8` | 1 |
| **columns total** | | **14** |

`pam_start` is `int32` rather than `int64` *because* of per-chromosome
sharding — coordinates are contig-local, and chr1 (248,956,422 bp) fits with
three orders of magnitude to spare. Sharding buys 4 B/site on top of its
streaming benefit.

**Mismatch counting by XOR/popcount.** For packed words `x`, `y`:

```
d = x ^ y
n_mm = popcount((d | (d >> 1)) & 0x5555...5555)
```

The fold collapses each 2-bit lane to one bit so `A↔T` (two flipped bits) and
`A↔C` (one) both count as exactly one mismatch. Uses `np.bitwise_count`
(numpy 2.3.5) across a whole candidate array at once.

**Counting-sorted offset arrays, not dicts.** Each seed block is CSR postings:
`offsets` (`uint64`, `n_buckets+1`) + `ids` (`uint32`, one entry per site),
built with `np.bincount` → `np.cumsum` → stable `argsort`. While a block is
≤ 8 nt the seed value directly addresses the bucket (no hashing, no probing);
wider blocks fall back to sorted keys + `searchsorted`. Postings are verified
in tests to be a true permutation of site ids, grouped and ascending.

**Vectorised CFD.** The published mismatch matrix is reshaped into a
`(20,4,4)` float table with a 1.0 diagonal, and scores accumulate over
positions 1→20 left-to-right — the same order as the scalar scorer, so results
are **bit-identical floats, not merely close**. The PAM factor is omitted only
because every indexed site is NGG and `pam_scores['GG'] == 1.0`.

---

## 2. Validation — all seven required checks

Oracle: the already-validated `SeedIndex`/`seed_index_search`, plus O(n)
`brute_force_search` as an independent ground truth. Hits are compared on
`(chrom, strand, pam_start, protospacer, pam, n_mm, cfd)` — coordinate keys,
never protospacer sequence alone (which aliases).

| # | Requirement | Result |
|---|---|---|
| 1 | Identical candidate positions | **PASS** — candidate id sets equal on every query |
| 2 | Identical mismatch counts | **PASS** — per-hit `n_mm` equal |
| 3 | Identical exact/1/2/3-mm counts | **PASS** — chrM tiers `{0:400, 1:138, 2:128, 3:135}` on both |
| 4 | Identical published CFD per-site scores | **PASS** — exact `==` and matching `float.hex()`, 1,101 scores |
| 5 | No false negatives | **PASS** — 180 brute-force cross-checks, 0 discrepancies |
| 6 | Memory usage measured | **PASS** — 12.3× reduction (table below) |
| 7 | Build/query timing measured | **PASS** — table below |

Coverage: the **entire chrM site set (all 2,193 guides)** queried exhaustively;
1,400 additional real/mutated/random queries; every mismatch budget
`max_mm ∈ {0,1,2,3}` (0 exercises the `searchsorted` wide-block path); 8 random
synthetic contigs; a low-complexity adversarial contig with duplicate
protospacers; a planted-off-target contig with known 0/1/2/3-mm loci (all
recovered, the 4- and 5-mm decoys correctly rejected); 5-shard streaming.

As established in the chrM stage, chrM has **zero** natural ≤3-mm off-target
pairs (min pairwise Hamming = 4), so off-target paths are driven by
deliberately mutated queries — expected, not a gap.

---

## 3. Memory (check 6)

Measured with `tracemalloc`. The packed shard replaces **both** the dict
postings **and** the resident list of `Site` dataclasses, so the honest
comparison is total resident footprint; the postings-only column is also given.

| dataset | sites | `Site` objs | dict postings | **dict total** | **packed** | **reduction** |
|---|---|---|---|---|---|---|
| chrM | 2,193 | 297.4 | 284.3 | **581.7 B/site** | **44.96 B/site** | **12.9×** |
| synthetic 400 kb | 49,972 | 298.1 | 81.9 | **379.9 B/site** | **30.66 B/site** | **12.4×** |
| synthetic 1.6 Mb | 200,719 | 297.6 | 73.0 | **370.6 B/site** | **30.16 B/site** | **12.3×** |

Asymptotic packed cost ≈ **30 B/site** = 14 B columns + ~16 B postings
(4 blocks × 4 B ids, plus a fixed 2×`(4^5+1)`×8 B offset table that amortises
away). chrM sits higher (45 B/site) only because that fixed table is spread
over just 2,193 sites — small-shard overhead, not the steady state.

**hg38 projection (arithmetic only — nothing downloaded):** at ~388 M NGG sites,
370.6 B/site → **~144 GB** for the current design vs 30.16 B/site → **~11.7 GB**
packed. That is the difference between "impossible on any normal machine" and
"fits in RAM on a large box, or trivially as 24 memory-mapped per-chromosome
shards." The original 110 GB estimate understated the problem because it
counted only the dict, not the `Site` objects.

## 4. Timing (check 7)

| dataset | dict build | packed build | dict/query | packed/query | query speedup |
|---|---|---|---|---|---|
| chrM (2,193) | 0.0281 s | **0.0022 s** (12.8×) | 25.9 µs | 158.0 µs | **0.16×** |
| 400 kb (49,972) | 0.4743 s | **0.0632 s** (7.5×) | 274.7 µs | 165.7 µs | **1.66×** |
| 1.6 Mb (200,719) | 1.9994 s | **0.3535 s** (5.7×) | 1245.7 µs | 278.4 µs | **4.47×** |

Build is 5.7–12.8× faster everywhere. Query has a **crossover**: on tiny shards
the fixed numpy dispatch cost (~150 µs) dominates and the packed index is
*slower* than plain Python dict lookups; by 200 k sites it is 4.5× faster and
the gap widens with shard size, because per-query work is vectorised while the
dict path scales with candidate count in interpreted code. This is the right
trade for genome scale — real hg38 shards are 10²–10³× larger than the largest
tested here — but it is worth stating plainly rather than reporting a
flattering average.

---

## 5. Honest limitations

- **Not yet tested above 200 k sites.** All numbers extrapolate to hg38 by
  arithmetic, not measurement. The projection assumes ~388 M sites.
- **Postings are still built in RAM per shard** (`argsort` over all sites in a
  contig). Fine per chromosome; a full-genome single-shard build would not be.
- **No on-disk/mmap serialisation yet.** Indexes are rebuilt in memory each run.
  This is the natural next step and is what makes the 11.7 GB figure practical.
- **NGG only.** NAG/other PAMs are not indexed, so the dropped PAM factor is
  safe today but must be restored if non-NGG sites are ever added.
- The `int32` coordinate choice asserts at build time; a contig > 2.1 Gb would
  raise rather than silently wrap.

---

## 6. Files

| path | role |
|---|---|
| `src/v2b/packed.py` | the packed index (new; only new file in `src/v2b/`) |
| `tests/test_packed_index.py` | 43 equivalence/structure/memory tests |
| `run_packed_validation.py` | stage runner |
| `reports/packed_index_report.json` | machine-readable results |

**Stopping here as instructed — no hg38 work has begun.**
