# V2B — Candidate-Filter Strategies: Theory + Small Benchmark

**Proposal stage. Nothing adopted, nothing modified, no hg38.** `packed.py`,
`index.py`, `search.py`, `store.py` all md5-unchanged; suite 197/197; workspace
981 KB. Scratch: `bench_filters.py`, `reports/filter_strategy_bench.json`.

---

## Headline

**Most of the Stage 5 query cost was never the filter.** Profiling the 5,111
µs/query at 10M sites first:

| stage | share of query |
|---|---|
| `np.unique` dedupe of candidate ids | **75.5 %** |
| `cfd_vectorised` | fixed ~120 µs, *independent of candidate count* |
| block lookups | 0.3 % |
| Hamming + gather | the rest |

Two of those are defects, not algorithmic costs, and both are fixable **without
any extra memory**. Combined with the best filter, at 2M sites:

| config | cand/query | µs/query | speedup | bit-identical |
|---|---|---|---|---|
| S0 today | 7,798 | 1,267.9 | 1.00× | — |
| S1 sort+diff dedupe | 7,798 | 398.6 | **3.18×** | yes |
| S1 + one-pass CFD | 7,798 | 258.8 | **4.90×** | yes |
| S2 pairs-10 | 305 | 191.2 | 6.63× | yes |
| **S2 + one-pass CFD** | 305 | **75.8** | **16.73×** | yes |

**Recommendation: do the two free fixes first (4.9× for ~20 lines and zero
memory), then decide on a filter.** I did not implement them — this is the
comparison you asked for.

---

## 1. The `np.unique` anomaly

`np.unique` costs **757 µs** where `sort` + `diff` on the same 8,000 ids costs
**32.6 µs** — 23× apart, and 70× on some shapes. cProfile shows why: numpy 2.3.5
routes through `_unique_hash`, which is **89 % of its runtime**, then sorts
anyway. The measured duplicate ratio is **1.00** — the four seed blocks are
disjoint by construction, so we were paying a full deduplication for
*essentially no duplicates*.

`fast_unique` (concatenate → `sort` → neighbour-compare) is a drop-in
replacement. **3.18× on its own, zero memory, zero format change.**

## 2. The CFD floor

A ~140 µs floor appears at **any** non-zero candidate count — even one
candidate. Breakdown at n=1: gather 0.9 µs, Hamming 6.1, lexsort 2.7,
**`cfd_vectorised` 120.5**. It loops 20 positions issuing 20 numpy ops on tiny
arrays; the cost is dispatch, not arithmetic.

A one-pass prototype (decode all 20 positions into one `(n,20)` matrix, single
`prod`) is **bit-identical** (`float.hex()` equality):

| n | current | one-pass | speedup |
|---|---|---|---|
| 1 | 120.2 µs | 19.5 | **6.08×** |
| 100 | 126.0 | 34.4 | 3.42× |
| 300 | 134.0 | 63.0 | 2.11× |
| 1,000 | 198.6 | 162.6 | 1.34× |
| 5,000 | 456.7 | 771.2 | **0.59× — slower** |

**It inverts above ~1,000 candidates** (the `(n,20)` matrix stops fitting cache).
So it is not a free win in isolation: it pays off *precisely because* a
selective filter keeps candidate sets small. The two changes are complements —
which is exactly why the combined row is 16.7× and neither part gets there alone.

## 3. The filter strategies

All use the same verification path, so only the filter differs.

| id | design | rule |
|---|---|---|
| S0/S1 | 4 × 5 nt blocks | ≥1 exact block (pigeonhole: 3 mm dirties ≤3 of 4) |
| S2 | 5 × 4 nt, all C(5,2)=10 pair tables on 8-nt keys | ≥2 exact blocks |
| S3 | 5 × 4 nt, ≥2 found by counting duplicates | ≥2, no extra tables |
| S4 | 6 blocks, 6 pair tables (minimum cover) | ≥2 exact blocks |

**Soundness.** With b blocks and m mismatches, ≥ b−m blocks are clean. b=5, m=3
→ ≥2 clean, and every 2-subset is a table. For b=6 a pair set fires on every
3-subset iff its complement graph is triangle-free; K₃,₃ has 9 edges (max
triangle-free on 6 vertices), so 15−9 = **6 tables suffice**. I verified this
exhaustively over all dirty-block subsets of size 0–3, and confirmed by search
that **no 5-pair cover works** — 6 is the true minimum.

Empirically: **0 false negatives** for S2/S3/S4 against exhaustive brute force
(150,018 sites, 300 queries at 0/1/2/3 mm, 301 true hits), and all five
strategies returned **bit-identical** hits/mismatches/CFD to the baseline.

### Theory vs measurement (2M sites)

Expected candidates = Σ over tables of n/4^(key nt):

| strategy | predicted | measured | filter memory |
|---|---|---|---|
| S0/S1 | 7,816 | 7,823 | 16.0 B/site |
| S2 pairs-10 | 305 | **304.9** | 42.6 B/site |
| S3 count-5 | 39,078 raw postings | 304.9 final | 20.0 B/site |
| S4 pairs-6 | 1,465 | 1,726 | 24.4 B/site |

Theory matches to <1 % for S0–S2. **S2 is 25.6× more selective than S0.**

### But selectivity ≠ proportional speed

| strategy | cand/q | filter µs | verify µs | ns/candidate |
|---|---|---|---|---|
| S1 | 7,808 | 60.8 | 356.3 | 46 |
| S2 | 302 | 24.1 | 159.0 | **526** |
| S4 | 1,727 | 23.4 | 186.8 | 108 |

S2 cuts candidates 26× but query time only ~2× (before the CFD fix), because the
fixed floor now dominates. **This is the key trade-off: past a certain point,
better filtering buys nothing until the floor is removed.** Hence the ordering
in the recommendation.

### Scaling

| sites | S1 cand | S2 cand | S1 µs/q | S2 µs/q |
|---|---|---|---|---|
| 499,529 | 1,951 | 76 | 247 | 184 |
| 1,000,142 | 3,902 | 153 | 279 | 178 |
| 2,000,986 | 7,796 | 301 | 376 | 195 |

Over 4× sites: S1 latency **1.52×**, S2 latency **1.06×** — near-flat. S2's
candidate growth is real but starts 26× lower, so the floor absorbs it over this
range. **This is the property that matters for hg38**, and the reason S2 is worth
considering despite the memory cost.

**The 32 Mbp run was OOM-killed** at 1.98 GB building 10 pair tables. That is a
result: S2's 42.6 B/site is 2.7× the baseline filter, and at hg38 scale
(388 M sites) it projects to **~16.5 GB of filter tables on top of the 11.7 GB
index**. S4 at 24.4 B/site (~9.5 GB) is the better memory/selectivity
compromise if a filter change is adopted at all.

## 4. Recommendation

1. **`fast_unique` dedupe** — 3.18×, ~10 lines, no memory, no format change.
2. **One-pass CFD** — 4.90× cumulative, bit-identical, no memory. Guard it with
   a size threshold (use the current loop above ~1,000 candidates).
3. **Then re-measure.** These two alone may put hg38 near ~40 ms/query
   (198 ms ÷ 4.9), which could be acceptable without touching the index format.
4. **Only if that is still too slow**, adopt a filter — **S4 pairs-6** for
   balance, or S2 for maximum selectivity if the memory is affordable. Both
   change the on-disk format and invalidate existing shards; steps 1–2 do not.

S3 is not worth it: same selectivity as S2 for half the memory, but it touches
39 k raw postings per query and measured *slower* than S1's dedupe fix.

## 5. Limitations

- Uniform-random synthetic sequence. **Real chromosomes have repeats that will
  skew candidate lists**, so all selectivity figures are optimistic averages;
  worst-case guides in repetitive regions will be worse, and repeats hurt the
  ≥2-block rules differently than the ≥1 rule.
- Largest corpus 2M sites (0.5 % of hg38); hg38 numbers are extrapolation.
- Single shard, single thread. Stage 5 measured a **~6 µs/shard fixed cost →
  ~143 µs/query floor across 24 hg38 shards**, which no filter change addresses.
- One-pass CFD is a ~15-line prototype, bit-identical on the shapes tested
  (n ≤ 5,000, 50 hex-compared values per shape), not a hardened implementation.
- Timings are single-run on a 2-core box; ±10 % run to run.

**Stopping here as instructed — comparison only, nothing adopted.**
