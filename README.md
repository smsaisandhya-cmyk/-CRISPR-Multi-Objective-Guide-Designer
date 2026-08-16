# CRISPR Guide RNA Design — Activity Prediction \& Off-Target Specificity

A two-part machine learning and algorithms project for CRISPR guide RNA design:

1. **Activity prediction** (V1 / V2A) — predict on-target cutting efficiency from
guide sequence, evaluated under a leakage-free gene-level split.
2. **Specificity engine** (V2B, *research extension*) — an exact sequence-based
off-target search with published CFD scoring, designed for genome-scale
indexing and validated on real human chrM and synthetic datasets up to
10 M sites.

Efficiency and specificity are deliberately kept as **separate components**: a
guide can cut well and still be unusable if it has close off-targets, so the two
scores are reported independently rather than blended.

> \*\*Status: V1/V2A application complete; V2B validated as a research extension.\*\*

> The current Streamlit application uses the V1 specificity backend. V2B is

> documented in `V2B\_RESEARCH/` and has not yet been integrated into the

> application. V2B figures below are measured validation results from the

> separate V2B development workspace; nothing is projected except where a line

> explicitly says "projection".

\---

## Results at a glance

### Activity prediction

|Model|Split|Spearman ρ|
|-|-|-|
|Random Forest|random guide split (V1)|0.728|
|CNN|random guide split (V1)|0.745|
|Random Forest|**gene-level split (V2A)**|**0.718**|
|CNN|**gene-level split (V2A)**|**0.720**|

The V2A numbers matter more than the V1 numbers. A random guide split lets guides
from the *same gene* appear in both train and test, which leaks. Re-splitting so
that **4,034 test genes are entirely unseen** (10,944 test guides) cost only
\~0.01–0.025 ρ — evidence the models learned sequence determinants rather than
memorising genes.

Dataset: DeepHF, 54,956 guides across 20,168 genes, gene labels joined from DeepHF
Supplementary Data 1 (632 unmatched + 16 ambiguous guides excluded).

### Specificity engine (V2B)

|Metric|Result|
|-|-|
|Correctness|**197/197 tests**; exact vs brute force, **0 false negatives**|
|Memory|**12.3× reduction** — 370.6 → 30.03 bytes/site|
|Query speed|**4.0× faster** after optimisation (4.87 ms → 1.22 ms at 10 M sites)|
|vs brute force|**222× faster** on human chrM (3.89 → 0.0175 ms/guide)|
|Scale validated|**10 M sites**, 300 MB index, linear build, flat bytes/site — hg38 was **projected, not executed**|
|Scoring|Published Doench 2016 CFD matrix, bit-identical to the reference scorer|

\---

## What this fixes from V1

The V1 specificity module had three defects that made its output unusable as a
real specificity estimate:

|V1 defect|V2B|
|-|-|
|`specificity\_proxy()` was a **heuristic fallback**, not a genome search|Real indexed genome search|
|Searched **only chr22**|Per-chromosome shards, designed for genome-scale indexing|
|**CFD-inspired position weights**, not the published matrix|Published Doench 2016 CFD tables|
|Reverse-strand bug: matched **every C** instead of true CCN|True CCN detection|

The reverse-strand bug was the most consequential: it over-called minus-strand
sites by **2.92×** on real chrM (5,170 spurious vs 1,769 true) and **3.93×** on
synthetic sequence. Every one of those phantom sites would have counted against a
guide's specificity score.

\---

## \## Quick start

## 

## The current deployable application is the Streamlit V1/V2A prototype.

## 

## From the project directory:

## 

## ```bash

## streamlit run app.py```

\---

## \## How it works

## 

## \### Current application

## 

## The deployed Streamlit application combines three main objectives:

## 

## \- \*\*Activity:\*\* CNN-based prediction of guide cutting efficiency.

## \- \*\*Structure:\*\* RNA secondary-structure analysis using ViennaRNA.

## \- \*\*Specificity:\*\* sequence-based off-target assessment using the V1 chromosome-22 backend.

## 

## The application then identifies \*\*Pareto-optimal guides\*\* rather than combining the objectives into a single weighted score.

## 

## \### V2B research extension

## 

## The V2B research extension explores a more rigorous specificity architecture based on:

## 

## \- true NGG and reverse-strand CCN detection

## \- published Doench 2016 CFD scoring

## \- seed-indexed candidate retrieval

## \- 2-bit sequence packing

## \- CSR-style postings

## \- memory-mapped chromosome shards

## 

## The V2B validation work demonstrated exact agreement with brute-force search on its validation datasets and explored scaling to synthetic datasets of up to 10 million sites.

## 

## The V2B implementation and benchmarks are preserved as a research record in `V2B\_RESEARCH/`. They are not currently part of the deployable Streamlit application.

## 

## Full design rationale and benchmark methodology: `ARCHITECTURE.md`.

## \---

## Components

|Stage|What it is|Status|
|-|-|-|
|V1|CNN activity prediction (DeepHF 55,604), chr22 CFD-inspired specificity, ViennaRNA structure, and Pareto ranking|✅ **Live in the Streamlit app**|
|V2A|Gene-level split evaluation (RF 0.718 / CNN 0.720, 4,034 held-out genes)|✅ **Complete**|
|V2B|Seed-index + published-CFD off-target specificity research prototype|⏳ **Validated research extension, not yet integrated into the app**|

**Note:** The deployed prototype currently uses the V1 specificity backend. V2A is an evaluation/validation stage rather than a separate app component. V2B is the next specificity upgrade and has been validated experimentally at the prototype level, but full genome-scale integration into the deployed application remains future work.

## Validation history

\### V2B validation record



The following results were obtained during V2B development in a separate validation workspace and are preserved here as a research record.



They should not be interpreted as executable tests in the current repository, because the complete V2B implementation and its test harness are not included in this application snapshot.
Each stage was gated on the previous one passing.

|Stage|What was proven|Result|
|-|-|-|
|1. Synthetic|PAM detection, pigeonhole index, CFD correctness|PASS, 88 tests|
|2. Real reference (chrM)|2,193 real sites, the N at 3107, repeats|PASS 5/5, 222×|
|3. Packed index|Identical results at 12.3× less memory|PASS 5/5, 162 tests|
|4. mmap store|Bit-identical round-trip, O(1) load|PASS 10/10, 197 tests|
|5. Scale stress|1/5/10 M sites; 30.03 B/site flat|PASS|
|6. Filter analysis|Theory + benchmark of 5 candidate strategies|Analysis|
|7. Optimisation|4.0× query speedup, zero format change|PASS|

The detailed V2B validation reports are preserved in `V2B\_RESEARCH/`.**Stage 5 found the real bottleneck.** Storage scales fine; *query time* is linear
in index size because candidates examined follow `4n/1024` (matched to within 1%).
Stage 6 compared five ways to cut that, and Stage 7 implemented the two that
required no format change — a sort+diff dedupe replacing `np.unique` (which spends
89% of its time in a hash path before sorting anyway) and a threshold-guarded
one-pass CFD kernel.

### Known limits

* Query latency is still **linear in index size**. Stage 7 cut the constant \~4×,
not the slope. hg38 projects to \~50 ms/query (from \~198 ms), plus a \~143 µs/query
fixed floor across 24 shards that no filter can remove.
* Going sub-linear needs a two-block-match filter (Stage 6 strategy S2, measured
16.7× end-to-end) — deliberately **not adopted**, because it costs \~16.5 GB extra
at hg38 scale and changes the on-disk format.
* **hg38 has never been run.** All scale figures use synthetic random sequence,
which lacks the repeat structure that will skew real candidate counts upward.
* Indels are not modelled; substitutions only.

\---

## Repository layout

```
## Repository layout



```text

app.py

src/

&#x20; baseline.py

&#x20; evaluate.py

&#x20; features.py

&#x20; multiobjective.py

&#x20; specificity.py

&#x20; v2\_map\_genes.py

&#x20; v2\_train\_eval.py



V2B\_RESEARCH/

&#x20; V2B\_CHRM\_QC\_REPORT.md

&#x20; V2B\_FILTER\_STRATEGY\_COMPARISON.md

&#x20; V2B\_HANDOFF\_RECORD.md

&#x20; V2B\_MMAP\_STORE\_REPORT.md

&#x20; V2B\_OPTIMISATION\_REPORT.md

&#x20; V2B\_PACKED\_INDEX\_REPORT.md

&#x20; V2B\_SCALE\_STRESS\_REPORT.md



ARCHITECTURE.md

V1\_EVALUATION\_PROTOCOL.md

V1\_vs\_V2A\_COMPARISON.md

requirements.txt
```

## \## Reproducibility

## 

## The V1/V2A application code and evaluation documentation are included in this repository.

## 

## V2B validation results were generated during the separate V2B development workspace and are preserved as research reports in `V2B\_RESEARCH/`. The complete V2B validation implementation is not included in this repository snapshot.

## 

## Large datasets, trained model files, genome references, and serialized indexes are intentionally excluded from version control.

## 

## Deliberate guardrails: no genome or serialized index is committed. Large-scale V2B tests used generated sequence and temporary scratch storage during validation. Total repository footprint is kept small by excluding large experimental artifacts.

