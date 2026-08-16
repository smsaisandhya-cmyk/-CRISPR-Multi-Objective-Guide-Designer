# V2B Stage 5 — Large-Scale Stress Test (1M / 5M / 10M sites)

Synthetic chromosomes only. **No hg38, nothing downloaded.** Workspace 949 KB;
all shards written to `/var/tmp` and deleted. `index.py`, `search.py`,
`packed.py`, `store.py` md5-unchanged; suite still **197/197**.

**The storage layer holds up perfectly — and the stress test found the real
ceiling somewhere else. Query time grows *linearly* with index size, and the
demand-paging advantage claimed in Stage 4 largely evaporates under sustained
querying.** Both are quantified below. Neither is a regression; both were
invisible at the 200 k sites tested previously.

Runner `run_scale_stress.py`, results `reports/scale_stress_report.json`.

---

## 1. Method

| Aspect | Choice, and why |
|---|---|
| Data | Deterministic per-contig `np.random.default_rng(SEED+i)`; 0.125 NGG sites/bp measured, so 8 Mbp ≈ 1 M sites |
| Shape | 1M = 2×4 Mbp, 5M = 5×8 Mbp, 10M = 10×8 Mbp — always multi-shard |
| RAM | Fresh subprocess per measurement, `/proc` `VmRSS` + `ru_maxrss`, **net of a measured interpreter+numpy baseline (~28 MB)** |
| Scratch | `/var/tmp` (real disk). **`/tmp` here is tmpfs — RAM-backed — so measuring "mmap RAM" there would be measuring nothing** |
| Cold cache | `posix_fadvise(POSIX_FADV_DONTNEED)` per shard; works without root (`/proc/sys/vm/drop_caches` is not writable) |
| Queries | 200 = 100 real protospacers + 50 mutants (1–3 mm) + 50 random controls |

The sandbox has **1,984 MB total RAM**. That is itself a finding: see §4.

## 2. Disk and build (measured)

| scale | sites | shards | disk | B/site | stream build | in-RAM build | CRC verify |
|---|---|---|---|---|---|---|---|
| 1M | 999,631 | 2 | 30.1 MB | 30.07 | 6.0 s | 5.9 s | ok |
| 5M | 5,001,140 | 5 | 150.2 MB | 30.03 | 28.5 s | 27.8 s | ok |
| 10M | 10,002,868 | 10 | 300.4 MB | 30.03 | 58.2 s | 55.7 s | ok |

**30.03 B/site, dead flat across a 10× range** and identical to the in-memory
figure — serialization costs nothing asymptotically. Build time is linear
(5.00×/2.00× sites → 4.74×/2.04× time). Disk scales exactly 5.00×/2.00×.

**Equivalence held at every scale**: 60 queries per scale re-run against the
mmap index returned identical chrom/strand/pam_start/mismatch tuples and
identical CFD scores compared via `float.hex()`. CRC valid on all 17 shards.

## 3. RAM (measured, net of baseline)

| scale | in-RAM build (resident) | build peak | mmap after load | mmap after 200 q | full copy | stream-build peak |
|---|---|---|---|---|---|---|
| 1M | 57.8 MB | 287.0 MB | **0.1 MB** | 26.2 MB | 42.9 MB | 271.9 MB |
| 5M | 200.2 MB | 660.7 MB | **0.0 MB** | 115.8 MB | 172.9 MB | 546.4 MB |
| 10M | 362.9 MB | 813.8 MB | **0.0 MB** | 232.6 MB | 315.9 MB | 549.8 MB |

Mapping a 300 MB index still costs **0.0 MB** and ~5 ms. But look along the
rows, not just at the load column.

### 3.1 The demand-paging claim needs qualifying

Stage 4 reported "0 KB resident after load" as the headline. At 200 k sites that
told the whole story. At scale it does not. I measured the paging curve directly
on a 150 MB index:

| queries | resident | share of file |
|---|---|---|
| 1 | 41.1 MB | **28.7 %** |
| 10 | 52.7 MB | 36.8 % |
| 50 | 85.5 MB | 59.7 % |
| 100 | 106.2 MB | 74.1 % |
| 400 | 124.7 MB | **87.0 %** |

**A single query faults in 28.7 % of the file, and 400 queries reach 87 %.**
The reason is structural, not incidental: a seed lookup gathers candidates
scattered across the whole `words` array, so it touches pages nearly everywhere.
mmap defers the cost; under a sustained query load it does not avoid it. The
honest claim is **"pay only for what you touch, and a busy server touches most
of it"** — mmap's true wins here are instant startup, sharing one page cache
across processes, and letting the kernel evict under pressure. It is not a
memory-footprint fix for a hot index.

Even so, at 10M: **363 MB resident to build vs 233 MB after 200 queries (1.6×)**,
and peak 814 MB vs ~233 MB (3.5×) — real, but far from Stage 4's 9.6–13.7×,
which was measured at a scale where paging had barely started.

## 4. The 10M in-RAM build nearly did not fit

Peak for the in-RAM build was **814 MB against 1,984 MB total** — and `Site`
objects cost **389 B each**, so materialising 10 M of them at once would need
~3.9 GB and would have been killed. The measurements above were only possible
because the build streams **one contig at a time**, freeing each site list after
packing. That is why stream-build peak plateaus at ~550 MB from 5M to 10M while
the in-RAM path keeps climbing. **Peak build memory is set by the largest single
contig, not by the index** — the property that makes per-chromosome hg38 sharding
viable.

## 5. Query behaviour — the actual ceiling

| scale | µs/query in-RAM | mmap warm | mmap cold | full copy | p50 | p95 | **candidates examined** |
|---|---|---|---|---|---|---|---|
| 1M | 638 | 702 | 970 | 716 | 678 | 1,156 | 3,882 |
| 5M | 2,597 | 2,966 | 4,786 | 3,499 | 2,814 | 3,583 | 19,524 |
| 10M | 5,111 | 5,651 | 8,633 | 5,637 | 5,562 | 6,380 | 39,025 |

Query cost is **linear in index size** (5M/1M = 4.07×, 10M/5M = 1.97×). It is not
a paging artefact — warm mmap, cold mmap and a full in-RAM copy all scale the
same way, and the in-RAM index is no better.

The cause is exact and predictable. With `max_mm=3` the guide splits into four
5-nt blocks = 1,024 buckets per block, so the expected candidate set is
`4n/1024`:

| scale | candidates measured | 4n/1024 predicted | ratio |
|---|---|---|---|
| 1M | 3,882 | 3,905 | 0.99 |
| 5M | 19,524 | 19,536 | 1.00 |
| 10M | 39,025 | 39,074 | 1.00 |

**Within 1 % at every scale.** Cost per candidate is constant (~131 ns) — the
vectorised Hamming/CFD kernels are fine. The seed filter simply is not selective
enough: at 10M sites it hands 39 k candidates to the verifier for every query.

### 5.1 What this means for hg38 (arithmetic, from measured rates)

| quantity | projection at 388 M sites |
|---|---|
| disk | **11.7 GB** (30.03 B/site — solid, measured flat over 10×) |
| stream build | ~38 min single-threaded, peak RAM set by chr1 (~30 M sites ≈ 0.9 GB) |
| in-RAM build | ~14.4 GB resident — **not viable on a 2 GB box; mmap is what makes it possible at all** |
| candidates/query | **~1.5 M** |
| **query latency** | **~198 ms** |

~198 ms/query is the finding that matters. It is usable for a single guide,
tolerable at 5 s for a 25-guide design, and unacceptable for genome-wide
all-against-all. **The pilot-scale conclusion "query speed is unchanged by
mmap" survives; the unstated hope that this design reaches hg38 interactively
does not.**

The fix is not more optimisation of the current path — it is a more selective
filter. `max_mm=3` over 20 nt with four blocks is the pigeonhole minimum; options
are longer seeds with more blocks (q-gram/(m+1)-block trade-off), restricting
exhaustive 3-mm search to a candidate gene set, or a two-stage filter that
requires two blocks to match. That is a design decision, not a tuning knob, so
I am stopping here as instructed rather than picking one.

## 6. Load time

| scale | open | map (warm) | map (cold) | full read (cold) |
|---|---|---|---|---|
| 1M | ~0.4 ms | 0.5 ms | 1.2 ms | 50 ms |
| 5M | ~0.5 ms | 1.2 ms | 3.3 ms | 177 ms |
| 10M | ~0.6 ms | 2.0 ms | 5.1 ms | 294 ms |

Mapping grows very slowly (2.31×/1.63× for 5×/2× data) because it is one `mmap`
call plus per-shard header parsing — it tracks *shard count*, not bytes. Full
copy is 60–150× slower and scales with bytes, as it must. Stage 4's "O(1) load"
is better stated as **O(number of shards)**.

## 7. Limitations

- Synthetic uniform-random sequence. Real chromosomes have repeats and low
  complexity that will make candidate lists **more** skewed — the 4n/1024 law is
  an optimistic average; worst-case guides in repetitive regions will be worse.
- Single-threaded throughout; no attempt at parallel query or build.
- 10M sites is 2.6 % of hg38. Disk and build extrapolate confidently (both flat
  and linear over 10×); the ~198 ms query projection assumes the candidate law
  keeps holding, which §5 supports but does not prove at 388 M.
- One machine, 2 cores, 1,984 MB RAM, no swap. Page-cache pressure behaviour on
  a box where the index far exceeds RAM is still untested — here 300 MB fit
  comfortably.
- Paging curve measured on one 150 MB index with one query mix; the exact
  percentages will shift with locality.

## 8. Files

| path | role |
|---|---|
| `run_scale_stress.py` | stress harness (subprocess RAM probes, cold-cache eviction) |
| `reports/scale_stress_report.json` | machine-readable results |

No production module was modified. **Stopping here as instructed.**
