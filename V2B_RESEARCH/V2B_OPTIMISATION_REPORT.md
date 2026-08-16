# V2B — Query-Path Optimisation (implementation report)

Scope as authorised: **only** the two zero-format-change optimizations.
No on-disk format change. S2/S4 not adopted. hg38 not downloaded. V1/V2A untouched.

| | |
|---|---|
| File changed | `src/v2b/packed.py` **only** (md5 `813da412…` → `a4cbc2a0…`) |
| Backup | `backups/packed.py.v1-pre-optimisation` (pre-edit, verified) |
| Other 8 modules | byte-identical, re-verified against `backups/MD5SUMS.pre-optimisation` |
| On-disk format | unchanged — **30.03 B/site**, identical to Stage 5 |

---

## 1. What changed

**(a) Candidate dedupe — new `sorted_unique()`, called from `PackedShard.candidates`.**
`np.unique(np.concatenate(parts))` → concatenate, `sort()`, neighbour-compare.
`np.unique` routes through `_unique_hash` (89% of its runtime by cProfile) and then
sorts anyway. Output is identical by construction: ascending, no repeats.

One correctness detail found during implementation: `BlockTable.lookup` returns a
**slice view into the CSR postings array**. Sorting that in place would corrupt the
index itself and would raise on a read-only mmap-backed array. The single-part
branch therefore copies. Verified: input array unmutated, read-only input accepted,
and **0 mismatches vs `np.unique` across 3,000 random shapes**.

**(b) One-pass CFD — threshold-guarded at `CFD_ONE_PASS_MAX = 1024`.**
`cfd_vectorised` now dispatches to `_cfd_one_pass` (decode all 20 positions into an
`(n,20)` array, single `.prod(axis=1)`) at or below the threshold, and to the
original `_cfd_loop` above it. The old loop is retained verbatim, so large batches
behave exactly as before.

I re-measured the crossover to place the threshold rather than inheriting the coarse
"between 1,000 and 2,000" bracket:

| candidates | loop µs | one-pass µs | ratio |
|---|---|---|---|
| 600 | 173.7 | 98.7 | 1.76× |
| 1000 | 184.3 | 149.9 | 1.23× |
| 1200 | 202.1 | 181.3 | 1.11× |
| **1400** | 212.5 | 316.0 | **0.67× (slower)** |
| 1800 | 243.2 | 408.6 | 0.60× |

True crossover is between 1,200 and 1,400; 1,024 sits below it with margin.

`.prod(axis=1)` reduces pairwise rather than left-to-right. That was checked before
implementing — the factors are exact table lookups, and across 116,669 scored values
it ties an explicit sequential loop on every `float.hex()`, while being faster.

## 2. Acceptance results

Old module (loaded from the backup as a parallel package) vs new module, same
synthetic data, both loading a **bitwise-identical CFD matrix**, compared on
`(chrom, strand, pam_start, n_mm, cfd.hex())`:

| chromosome | sites | candidates/query | queries above threshold |
|---|---|---|---|
| 0.2 Mbp | 25,213 | 74–131 | 0 / 120 |
| 1.0 Mbp | 124,900 | 430–537 | 0 / 120 |
| 4.0 Mbp | 499,425 | 1,778–2,057 | **120 / 120** |

**361 hit rows compared, 0 mismatches** — and both sides of the threshold are
genuinely exercised, so the dispatch itself is covered, not just the fast path.

- **Brute force** (150,148 sites, 450 queries, full Hamming scan + published scalar
  `cfd_score`): 451 true hits, **0 false negatives, 0 spurious**.
- **Unit tests: 197/197 pass.**

## 3. Benchmark at existing scale (fresh subprocess per point)

| sites | index | old µs/q | **new µs/q** | speedup | cand/q | build s | peak RSS |
|---|---|---|---|---|---|---|---|
| 1 M | 30.1 MB | 657.9 | **175.8** | **3.74×** | 3,894 | 9.8 → 10.5 | 541 MB |
| 5 M | 150.2 MB | 2,600.8 | **633.6** | **4.11×** | 19,470 | 33.5 → 33.2 | 679 MB |
| 10 M | 300.4 MB | 4,866.8 | **1,221.1** | **3.99×** | 38,953 | 60.0 → 60.7 | 832 MB |

- **Speedup 3.7–4.1×** on query latency; 10 M drops 4.87 ms → 1.22 ms.
- **RAM and index size unchanged** (30.03 B/site; RSS differs by <0.5 MB — within noise).
- **Build time unchanged** (both paths are query-side only).
- Candidate counts identical old vs new, as required — this is pure per-candidate
  cost reduction, not filtering.

Slightly below the 4.9× projected from the 2 M-site Stage 6 run: at 10 M most queries
carry ~39 k candidates, far above the threshold, so CFD keeps the original loop and
only the dedupe fix applies. That is the intended, conservative behaviour.

## 4. Honest limits

- **This does not change the scaling law.** Candidates still grow as `4n/1024`;
  latency is still linear in index size. The constant shrank ~4×, the slope did not.
- Extrapolating to hg38 (~388 M sites) gives roughly **50 ms/query** versus ~198 ms
  before, plus the ~143 µs/query fixed shard floor. Better, but still linear —
  a real fix needs a candidate-count reduction (S2/S4), which was explicitly not adopted.
- All figures are synthetic random sequence. Real genomes have repeats, which will
  skew candidate counts above the uniform-random model.

Stopping here as instructed — no hg38 work.
