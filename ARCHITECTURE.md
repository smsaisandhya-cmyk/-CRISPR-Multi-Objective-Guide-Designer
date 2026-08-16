# V2B Architecture & Design Notes

Technical companion to the README: the index format, why the search is exact, the
optimisation history, and the benchmark methodology. Written for a reviewer who
wants to check the claims rather than take them on faith.

---

## 1. Problem

Given a 20-nt guide, find every site in a genome within `max_mm = 3` mismatches of
it that is adjacent to an NGG PAM, and score each by the published CFD model. At
hg38 scale that is ~388 M candidate sites. Brute force is O(genome) per guide with
a large constant; the naive dict-of-lists index is ~144 GB of RAM. Neither is usable.

Two properties are non-negotiable:

- **Exactness.** A missed off-target is a potentially unsafe guide. Any candidate
  filter must be proven to have zero false negatives, not measured to be "close".
- **Bit-identical scoring.** CFD values must match the published scorer exactly,
  compared by `float.hex()` rather than a tolerance, so that any future refactor
  that perturbs floating-point ordering is caught immediately.

## 2. Why the search is exact — the pigeonhole guarantee

Split the 20-nt protospacer into `max_mm + 1 = 4` contiguous blocks of 5 nt. A true
off-target has at most 3 mismatches distributed among 4 blocks, so **at least one
block must be mismatch-free**. Index every site under all four of its block values;
for a query, look up all four block values and union the results. Any site within 3
mismatches is guaranteed to appear in that union.

This makes the seed index a *lossless accelerator*, not an approximation: it is a
candidate generator that provably over-generates, followed by an exact verification
pass (`hamming_packed`) that removes the extras. It is a filter with a proof, not a
heuristic with a threshold.

The guarantee is also verified empirically at every stage rather than assumed —
brute-force equivalence on synthetic sequence, on all 2,193 real chrM guides, and
again after the Stage 7 optimisation (450 queries, 451 true hits, 0 false negatives).

**Block geometry.** `block_bounds(20, max_mm+1)` at `max_mm=3` gives four 5-nt
blocks = 10 bits = 1,024 buckets per block, 8,200 B offset table per block. The
expected candidate count is `Σ n / 4^(block nt)` = `4n/1024`, which measurement
matched to within 1%.

## 3. Index format

### Layer 1 — 2-bit packing

A 20-mer packs into a single `uint64`, MSB-first, 2 bits per base. Mismatch counting
is then branch-free:

```
d = a XOR b
popcount((d | d >> 1) & 0x5555555555555555)
```

The `(d | d>>1)` fold collapses each 2-bit lane to a single "differs" bit, so one
popcount yields the Hamming distance in bases. Replaces per-character comparison.

### Layer 2 — CSR postings

Rather than `dict[block_value] -> list[site_id]` (huge per-object overhead), each
block table is a counting-sorted CSR pair: a `uint64` offsets array indexed by block
value, and a flat `uint32` ids array. Lookup is a slice, not a hash. When the key
space is small enough (`n_bits <= DIRECT_ADDRESS_MAX_BITS = 16`) the offsets array is
directly addressed; above that it falls back to `searchsorted` on a sparse key list.

This is where most of the 12.3× memory win comes from: **370.6 → 30.03 bytes/site**.

### Layer 3 — on-disk `.v2bs` shards

One file per chromosome: 24-byte preamble, JSON header, then 64-byte-aligned arrays.
CRC32 over the payload; writes are atomic (temp file + rename). Loading is
`np.frombuffer` views over an `mmap` — no copy, no pickle, no parse.

Load is **O(1)**: ~0.3 ms whether the shard holds 2 k or 200 k sites, versus
1.8–230 ms to rebuild. Coordinates are `int32`, which caps a contig at 2^31 bp —
fine for any human chromosome.

**Sharding is what makes hg38 feasible at all.** Peak build RAM is set by the largest
single contig, not the genome, because contigs stream one at a time.

## 4. CFD scoring

Tables are the published Doench 2016 mismatch matrix and PAM scores, re-derived from
the CRISPOR reference implementation (`CFD_Scoring/cfd-score-calculator.py`).

Two fidelity details worth recording:

- The upstream scorer is **Python 2**; it is reimplemented, not imported. The
  reimplementation is checked against hand-computed values and a Python 3 port's
  doctest values.
- The mismatch table is stored as a `(20, 4, 4)` float array with a **1.0 diagonal**,
  so a matching base contributes a no-op factor and no branch is needed per position.
- Every indexed site carries an NGG PAM by construction, so the PAM factor is always
  `pam['GG'] == 1.0` and is dropped from the hot loop.

A counter-intuitive property that broke an early assumption: **seed-region mismatches
do not always cost more than PAM-distal ones**. `rG:dT` scores 0.9 at position 1 but
0.9375 at position 20. Any "weight the seed higher" shortcut is wrong; use the table.

## 5. Optimisation history (Stage 7)

Stage 5 established that **query time, not storage, is the ceiling** — candidates
grow as `4n/1024`, so latency is linear in index size. Profiling the 5,111 µs/query
at 10 M sites attributed:

| component | share |
|---|---|
| `np.unique` on the candidate list | **75.5%** |
| `cfd_vectorised` fixed overhead | ~120 µs/call regardless of n |
| block lookups | 0.3% |

Two fixes were applied; neither changes the on-disk format.

**(a) `sorted_unique` replacing `np.unique`.** On numpy 2.3.5 `np.unique` routes
through a `_unique_hash` path that cProfile attributes 89% of its runtime to — and
which then sorts anyway. Concatenate + `sort()` + neighbour-compare is 23× faster on
the same 8,000 ids (757 µs → 32.6 µs) and produces an identical array.

> **Implementation trap.** `BlockTable.lookup` returns a *slice view into the CSR ids
> array*. Sorting it in place would corrupt the index itself, and would raise on a
> read-only mmap-backed array. The single-part branch must copy. Verified: input
> unmutated, read-only input accepted, 0 mismatches vs `np.unique` over 3,000 random
> shapes.

**(b) Threshold-guarded one-pass CFD.** The original kernel does one numpy pass per
position — 20 dispatches whose fixed cost (~120 µs) dominates short candidate lists.
The one-pass version decodes all 20 positions into an `(n, 20)` array and does a
single `.prod(axis=1)`.

This is only faster while the intermediate stays cache-resident, so it is dispatched
by a measured threshold, `CFD_ONE_PASS_MAX = 1024`:

| candidates | per-position loop | one-pass | ratio |
|---|---|---|---|
| 600 | 173.7 µs | 98.7 µs | 1.76× |
| 1000 | 184.3 µs | 149.9 µs | 1.23× |
| 1200 | 202.1 µs | 181.3 µs | 1.11× |
| **1400** | 212.5 µs | 316.0 µs | **0.67× (slower)** |
| 1800 | 243.2 µs | 408.6 µs | 0.60× |

Crossover is between 1,200 and 1,400; the threshold sits below it with margin, and
the original loop is retained verbatim above it.

> **Float-ordering check.** `.prod(axis=1)` reduces *pairwise*, not left-to-right,
> so it is not obviously bit-identical to the sequential scalar loop. This was
> verified before adopting it: across 116,669 scored values it ties an explicit
> sequential loop on **every** `float.hex()`. The factors are exact table lookups,
> which is why the reassociation is safe here — this would not be safe in general.

## 6. Candidate-filter analysis (Stage 6) — considered and rejected

Five strategies were compared theoretically and benchmarked on ~2 M sites, 120
queries. All five produced bit-identical results.

| strategy | bytes/site | cand/query | µs/query |
|---|---|---|---|
| S0 baseline (4×5 nt, `np.unique`) | 16.0 | 7,823 | 1,132.5 |
| S1 sort+diff dedupe | 16.0 | 7,823 | 435.3 |
| S2 5×4 nt, ≥2 blocks, 10 pair tables | 42.6 | 304.9 | 142.9 |
| S3 counting filter | 20.0 | 304.9 | 456.1 |
| S4 6 blocks, 6 pair tables | 24.4 | 1,726 | 172.1 |

**Soundness was proven, not assumed.** For a two-block-match filter the question is
whether the indexed pair set fires on *every* possible distribution of `m` mismatches
across `b` blocks. Exhaustive enumeration over all dirty-block subsets of size 0–3
established the rule: **a pair set is sound iff its complement graph is triangle-free**.
S2 (all 10 pairs) and S4 (pairs within `{0,1,2} ∪ {3,4,5}`) are sound; no 5-pair cover
on 6 blocks is sound, so 6 pair tables is the minimum. Confirmed empirically: 150,018
sites × 300 queries, 0 false negatives.

S2 combined with the one-pass CFD reached **16.7× end-to-end**. It was **not adopted**:
it adds ~16.5 GB at hg38 scale (S4 ~9.5 GB) and changes the on-disk format. The
project was frozen at the two zero-cost wins instead — a deliberate scope decision,
and the single most likely thing a reviewer would ask about.

## 7. Benchmark methodology

Practices adopted after specific failures:

- **RAM is measured in a fresh subprocess** minus a measured baseline. `ru_maxrss`
  deltas inside the benchmark process are meaningless once numpy has warmed up.
- **Scratch lives on real disk, never `/tmp`.** `/tmp` is tmpfs (RAM-backed), which
  makes every mmap residency figure a fiction.
- **Cache is dropped with `posix_fadvise(POSIX_FADV_DONTNEED)`**, since
  `/proc/sys/vm/drop_caches` is not writable as a normal user.
- **Queries are real protospacers plus mutated variants.** Random 20-mers essentially
  never fall within 3 mismatches of a random contig, so a random-query benchmark
  measures the empty path and reports a flattering number.
- **Bytes-per-site is only measured on ≥400 kb shards**; on tiny shards fixed
  overhead dominates and the figure is meaningless.
- **Memory baselines count `Site` objects + postings**, not dict postings alone —
  the latter understates the true baseline ~3×.

Honest caveats that survive into the final numbers:

- mmap does **not** mean "~0 RAM for a hot index". One query faults in 28.7% of a
  150 MB index; 400 queries reach 87%. It buys instant startup and kernel-managed
  eviction.
- The packed index is **not** uniformly faster: it is 0.16× on chrM and only ≥1×
  past ~50 k sites. It is a scale optimisation, and small inputs pay for it.
- Linear query growth is **not** a paging artifact — warm mmap, cold mmap and a
  pure in-RAM copy all scale identically. The cause is the candidate law.

## 8. Measured scale (synthetic, no hg38)

| sites | index | build | query before | query after | speedup |
|---|---|---|---|---|---|
| 1 M | 30.1 MB | 9.8 s | 657.9 µs | 175.8 µs | 3.74× |
| 5 M | 150.2 MB | 33.5 s | 2,600.8 µs | 633.6 µs | 4.11× |
| 10 M | 300.4 MB | 60.0 s | 4,866.8 µs | 1,221.1 µs | 3.99× |

Bytes/site is flat at 30.03 across the whole range; build time is linear; candidate
counts are unchanged by the optimisation (it reduces per-candidate cost, not the
candidate set).

**hg38 projection — arithmetic only, never executed.** ~388 M sites → 11.7 GB on
disk, ~38 min streaming build, ~50 ms/query (from ~198 ms pre-optimisation), plus a
~143 µs/query fixed floor across 24 shards. Synthetic sequence lacks repeat
structure, so real candidate counts will run higher than the uniform-random model.
