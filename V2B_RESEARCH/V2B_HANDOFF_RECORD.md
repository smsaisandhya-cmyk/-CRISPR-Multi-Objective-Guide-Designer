# V2B Handoff Record

**Purpose:** Authoritative record reconciling the transferred-chat V2B validation
work against the code actually present in this canonical workspace.
Created: 2026-08-16 (post-reconciliation).
Status: Record only. No fabrication of missing implementation files.

---

## 1. What V2B was validated in the transferred workspace

The transferred chat completed (and validated) the following V2B work. This is the
authoritative **validation record** for everything after V2A:

- **Reverse-strand bug fix (V1 → V2B):** V1's reverse-strand scan required only
  `arr[i] == C` instead of a true **CCN** dinucleotide, causing substantial false
  reverse-strand sites. V2B requires `arr[i] == C AND arr[i+1] == C`. Validated
  quantitatively on synthetic and real human mtDNA.
- **Published CFD scoring:** canonical `mismatch_score.pkl` (20 positions × 12
  mismatch types = 240 values) + `pam_scores.pkl`. Reproduces reference CFD
  calculations exactly, including the T→U / mismatch-key convention.
- **Seed-index algorithm:** PAM-compatible NGG / reverse CCN site detection →
  20-nt protospacer extraction → PAM-proximal seed candidate generation → full
  20-mer verification → keep ≤3-mismatch sites → deduplicate genomic positions →
  per-site published CFD → project-specific aggregated specificity.
- **Brute-force equivalence:** identical candidate positions, mismatch counts,
  0/1/2/3-mismatch tiers, and CFD values; zero false negatives.
- **Synthetic validation suite: 197/197 tests passing** (NGG detection, true CCN
  detection, reverse-complement handling, seed indexing, brute-force equivalence,
  exact/1/2/3-mismatch detection, published CFD scoring, manual CFD calculations,
  packed indexing, mmap storage, scale stress testing, optimisation equivalence).
- **Real human sequence validation:** NC_012920.1 / rCRS mitochondrial genome
  (16,569 bp, one N position, 2,193 valid PAM-flanked sites: 424 forward NGG,
  1,769 reverse CCN). All chrM tests passed; index and brute-force agreed; the
  N-containing region handled correctly; V1 reverse-strand bug independently shown
  to over-call sites on real mtDNA.
- **Packed index:** measured ~30.03 bytes/site; verified bit-identical results vs
  the validated implementation.
- **mmap / on-disk storage layer:** O(1)-style index loading, demand paging, shard
  support, CRC validation; query results bit-identical. **Documented limitation:**
  mmap does not eliminate memory/page-cache costs at genome scale.
- **Scale stress tests:** 1M / 5M / 10M sites. Index scaled cleanly (~30.03
  bytes/site, ~linear build time, fast mmap loading, bit-identical results).
  Query candidate counts increased approximately linearly; projection
  `candidate count ≈ 4n/1024`. A full hg38 build would require further
  optimisation for interactive use. **hg38 was NOT downloaded or built.**
- **Query optimisation (final):** two zero-format-change optimisations
  (sorted/diff dedup path; one-pass CFD for small candidate sets; `BlockTable
  .lookup()` view-copy safety fix). Fresh subprocess benchmarks:

  | Sites | Old µs/query | New µs/query | Speedup |
  | ----: | -----------: | -----------: | ------: |
  |    1M |        657.9 |       175.8  |  3.74×  |
  |    5M |       2600.8 |       633.6  |  4.11×  |
  |   10M |       4866.8 |      1221.1  |  3.99×  |

  Acceptance: 361 old/new hit rows compared, 0 mismatches, CFD bit-identical;
  451 brute-force true hits, 0 false negatives, 0 spurious hits; 197/197
  regression tests pass; index format unchanged; 30.03 bytes/site unchanged.

---

## 2. What V2B files are actually present in this canonical workspace

- `src/v2b_specificity.py` — the **executable V2B pilot/reference**: true CCN
  detection, embedded published CFD matrix (240 entries) + PAM scores, seed-index
  search (`score_guide_index`), brute-force reference (`score_guide_brute`),
  summarizer (`summarize`), and test commands (`--cfd-check`, `--synthetic`,
  `--equivalence`, `--known-targets`, `--bench`, `--build`, `--score`).
- `results/v2b/synth.fa.gz` — tiny synthetic validation genome.
- `results/v2b/index/synth.npz` — tiny synthetic index (test artifact).
- `cfd/mismatch_score.pkl`, `cfd/pam_scores.pkl`, `cfd/README.txt` — canonical
  published CFD data source (present and verified: 240 matrix entries).

---

## 3. Which post-transfer implementation artifacts are missing

The following were validated in the transferred workspace but their **code/files
are NOT present** in this canonical workspace (verified by inspection; not
fabricated):

- Packed-index implementation (as a separate module/file) with ~30.03 bytes/site
  footprint.
- mmap / on-disk storage layer (`BlockTable`, shards, CRC validation).
- Query-optimisation code paths (sorted/diff dedup; one-pass CFD threshold).
- 197-test regression suite module.
- 1M / 5M / 10M scale stress-test scripts + outputs.
- chrM (NC_012920.1) validation script + outputs.
- V2B QC report / benchmarks file.

---

## 4. Distinction between validated results and code currently present

- **Validated results (authoritative):** everything in §1 — the algorithm design,
  correctness guarantees (equivalence, 197/197, chrM, CFD exactness), the packed
  footprint, the mmap findings + limitation, and the ~4× optimisation benchmarks.
  These are accepted as the project's V2B validation record.
- **Code present (this workspace):** only the executable pilot in §2. The pilot
  implements the same core algorithm (CCN fix + published CFD + seed-index +
  brute-force equivalence + synthetic tests) and is the reference executable.
- **Missing (not reconstructed):** the later storage/optimisation layer described
  in §3. It is documented here as validated-but-not-present. No attempt is made
  to rebuild it from memory; doing so would risk fabricating unverified code.

---

## 5. Current status / decisions (do not modify)

- V1: frozen reference. Do not modify.
- V2A: complete and locked (mapping + gene-level eval: RF 0.718 / CNN 0.720).
- V2B: handoff accepted as validation record; `src/v2b_specificity.py` kept as the
  executable pilot/reference; missing artifacts documented, not fabricated.
- Streamlit interface: **initially keep the existing V1 chr22 specificity
  backend. Do NOT integrate V2B into the app yet.**
- Do not download hg38. Do not run new V2B experiments/optimisations.
- Next step (after this record): report status; UI work will follow on explicit
  instruction.
