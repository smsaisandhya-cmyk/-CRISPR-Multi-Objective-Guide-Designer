# V2B Stage 4 — On-Disk / Memory-Mapped Chromosome Shards

**Result: PASS** — 10/10 validation groups, 35 new tests, **197/197** total suite.
A serialized, memory-mapped index returns **bit-identical** results to the
in-memory packed index and to the original `SeedIndex` oracle, loads in
**constant time (~0.3 ms)** regardless of size, and needs **9.6–13.7× less RAM**
than rebuilding.

No hg38 downloaded or built. `index.py`, `search.py` and `packed.py` are
byte-identical to their approved versions (md5 verified); V1, V2A and `app.py`
untouched. Machine-readable results: `reports/mmap_store_report.json`.

---

## 1. Format

New module `src/v2b/store.py`. One self-describing file per chromosome,
`<chrom>.v2bs`:

```
0    magic b"V2BSHARD" | version u32 | flags u32 | hdr_len u32 | reserved u32
24   JSON header (array name -> dtype, shape, absolute offset)
     zero padding
D    array payload, every array 64-byte aligned
```

The header records each array's dtype, shape and absolute offset, so loading is
`np.frombuffer(mm, dtype, count, offset)` — **a view into the mapping, no copy
and no parsing of the payload**. 64-byte alignment keeps every view naturally
aligned (numpy's fast paths on `uint64` require it) and cache-line friendly.

Deliberate choices:

- **No pickle.** Raw arrays + JSON only; a test asserts the pickle protocol
  marker never appears. A shard file is inspectable and cannot execute code.
- **CRC32 over the data region**, checked *on demand* rather than at open —
  verifying touches every page and would defeat lazy loading. `verify=True`
  opts in; `verify_shard()` / `verify_all()` check explicitly.
- **Atomic writes** via `tmp` + `os.replace`, so a crash never leaves a
  half-written shard observable.
- **Endianness recorded and enforced** — a big-endian file is rejected rather
  than silently returning byte-swapped garbage.
- **The loaded object is a genuine `PackedShard`** whose arrays happen to be
  mmap views, and `MappedIndex` subclasses `PackedIndex` and inherits `search()`
  verbatim. Result equality is therefore **structural — the same code path —
  not a coincidence that testing has to chase.**

---

## 2. Validation

Every check ran against **both** the in-memory packed index **and** the
validated `SeedIndex` oracle, in mmap and full-copy modes.

| Requirement | Result |
|---|---|
| 1. Serialize the validated packed index | **PASS** — 1 file/chromosome + manifest |
| 2. Load back via mmap / equivalent | **PASS** — zero-copy views; full-copy mode also provided |
| 3. Bit-identical to in-memory | **PASS** — all 2,193 chrM guides exhaustively |
| 4. Candidate positions | **PASS** |
| 4. Mismatch counts | **PASS** |
| 4. Exact/1/2/3-mm tier counts | **PASS** — chrM `{0:300, 1:110, 2:104, 3:86}` on both |
| 4. Published CFD scores | **PASS** — exact `==` and matching `float.hex()` |
| 5. Disk / RAM / build / query measured | **PASS** — tables below |
| 6. Multiple shards | **PASS** — 8 contigs, all 8 returning hits, all CRCs valid |
| 7. Workspace below 128 MB | **PASS** — **909 KB**; temp indexes written to `/tmp` and deleted |

Also covered: no false negatives vs O(n) `brute_force_search`; every
`max_mm ∈ {0,1,2,3}`; corruption detection (one flipped bit → CRC fails →
`ShardFormatError`); foreign/truncated file rejection; read-only enforcement on
mmap views; lazy loading (`len()` answered from the manifest with **0** shards
open); reopening leaves files byte-identical.

---

## 3. Disk size (check 5)

| dataset | sites | in-memory | on disk | B/site | format overhead |
|---|---|---|---|---|---|
| chrM | 2,193 | 98,590 B | 100,676 B | 45.91 | 2.07 % |
| synthetic 400 kb | 49,972 | 1,531,960 B | 1,533,904 B | 30.70 | 0.13 % |
| synthetic 1.6 Mb | 200,719 | 6,054,370 B | 6,056,252 B | 30.17 | **0.03 %** |

On-disk size ≈ in-memory size; the delta is header + alignment padding only,
and it amortises to negligible. chrM's higher B/site is the fixed per-block
offset table (~8.2 KB × 4) spread over few sites — the small-shard overhead
documented in Stage 3, not a format cost.

## 4. RAM (check 5)

Measured in **clean subprocesses** reading `VmRSS` from `/proc`, with the
interpreter+numpy baseline (~28 MB) subtracted. *In-process `ru_maxrss` was
tried first and reported a misleading 0 KB, because the build had already
pushed the high-water mark up — the figures below are net of that.*

| dataset | build in RAM | mmap after load | mmap after 300 queries | full-copy | **reduction** |
|---|---|---|---|---|---|
| synthetic 400 kb | 22,216 KB | **0 KB** | 2,324 KB | 1,492 KB | **9.6×** |
| synthetic 1.6 Mb | 81,272 KB | **116 KB** | 5,944 KB | 6,124 KB | **13.7×** |

**~0 KB resident immediately after mapping** is the headline: pages arrive only
as queries touch them. After 300 queries the resident set is still a fraction of
the file, because a seed lookup touches a few postings pages, not the whole
index. Full-copy mode pays the whole file up front, as expected — it exists as a
baseline and for filesystems where mmap is undesirable.

## 5. Timing (check 5)

| dataset | build | write | CRC verify | open | map all shards | full read | µs/query mmap | µs/query in-RAM |
|---|---|---|---|---|---|---|---|---|
| chrM | 1.8 ms | 0.6 ms | 0.1 ms | 0.24 ms | **0.38 ms** | 0.4 ms | 139.7 | 141.2 |
| 400 kb | 46 ms | 2.6 ms | 1.5 ms | 0.22 ms | **0.31 ms** | 1.0 ms | 162.5 | 178.7 |
| 1.6 Mb | 230 ms | 6.0 ms | 3.2 ms | 0.23 ms | **0.62 ms** | 2.8 ms | 229.2 | 266.6 |

**Mapping is O(1):** ~0.3 ms whether the shard holds 2 k or 200 k sites, while
rebuilding grows linearly (1.8 ms → 230 ms, a 128× spread). That is the whole
point — a genome index becomes a startup cost of milliseconds instead of
minutes.

**Query speed is unchanged** (within ±16 % run-to-run noise, sometimes faster
than in-RAM because mapped pages come from page cache). Memory mapping costs
nothing at query time.

> An earlier gate here read "mmap load ≥10× faster than rebuild" and **failed on
> chrM at 7.2×** — a bad gate, not a bad result: chrM rebuilds in 2 ms, so the
> ratio is meaningless at that size. It was replaced with the property that
> actually matters (map time constant and always faster than rebuild). Two other
> gates were also corrected after they passed for the wrong reasons; see §7.

---

## 6. hg38 projection (arithmetic only — nothing downloaded)

At ~388 M NGG sites and 30.17 B/site: **~11.7 GB on disk**, built once as ~24
per-chromosome shards, mapped in ~0.3 ms per shard, with resident RAM
proportional to the pages actually touched rather than to 11.7 GB. Against the
original dict-of-lists design's ~144 GB of anonymous RAM, this is the difference
between impossible and routine.

## 7. Bugs and mis-specified gates found (and fixed)

Recording these because each one initially produced a *green or plausible*
result that was wrong:

1. **`mmap.close()` raised `BufferError`** while numpy views were still
   exported. Closing anyway would have been a use-after-free; the code now drops
   the mapping reference and lets CPython unmap once the last view dies, always
   releasing the file handle.
2. **`load_all()` had a stray early `return`** that opened only the first shard.
3. **The RAM gate passed vacuously** — its filter produced an empty list, and
   `all([])` is `True`. It now requires ≥2 measured datasets.
4. **RAM deltas read 0 KB** (`ru_maxrss` baseline already at high water) and then
   **~28 MB of interpreter overhead** was being counted as index memory. Both
   corrected; §4 is net of an explicitly measured baseline.
5. **Three test assumptions were wrong, not the code**: `max_mm=1` also uses the
   sparse-keys path (two 10-nt blocks = 20 bits > the 16-bit direct cap); random
   20-mers essentially never land within 3 mm of a random 6 kb contig, so
   multi-shard coverage needs seeded/mutated guides; and bytes-per-site is not a
   meaningful assertion on tiny shards.
6. **A benchmark probe re-seeded `random.Random(98)` inside the generator**,
   producing a homopolymer and 1.6 M "sites" from 1.6 Mb (8× too many). Caught
   because the site count was implausible.

## 8. Honest limitations

- **Still nothing measured above 200 k sites.** The 11.7 GB hg38 figure is
  arithmetic. Page-cache behaviour at genome scale — where the index far exceeds
  RAM — is genuinely untested and is the main open risk.
- **Writing still requires building the shard in RAM first.** Per chromosome
  that is fine (chr1 ≈ 30 M sites ≈ 0.9 GB); there is no streaming writer yet.
- **No compression.** Postings would compress well, at a CPU cost per query.
- **Format is little-endian only** (enforced, not assumed) and **v1** — there is
  no migration path yet for future layout changes beyond the version check.
- **CRC is not checked by default on open.** Silent bit-rot would go unnoticed
  unless `verify=True` or `verify_all()` is used.

## 9. Files

| path | role |
|---|---|
| `src/v2b/store.py` | serialization + mmap loading (new; only new file in `src/v2b/`) |
| `tests/test_mmap_store.py` | 35 format/equivalence/laziness/corruption tests |
| `run_mmap_validation.py` | stage runner |
| `reports/mmap_store_report.json` | machine-readable results |

**Stopping here as instructed — no hg38 work has begun.**
