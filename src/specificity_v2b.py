"""
specificity_v2b.py -- V2B-powered off-target scan (published Doench 2016 CFD + true CCN).

RECONSTRUCTION PROVENANCE (honesty first)
-----------------------------------------
The original V2B pilot (src/v2b_specificity.py) was validated in a separate
development workspace: 197/197 synthetic tests, zero false negatives vs brute
force, ~4x speedup, chrM pilot 2,193 sites (424 NGG fwd / 1,769 CCN rev).
That workspace was not preserved; the validation RECORD lives on in
V2B_RESEARCH/*.md. This module re-implements the same documented algorithm
(V2B_HANDOFF_RECORD.md section 2):

    PAM-compatible NGG / reverse true-CCN site detection
    -> 20-nt protospacer extraction
    -> full 20-mer verification, keep <=3-mismatch sites
    -> deduplicate genomic positions
    -> per-site published CFD  ->  project aggregate  1 / (1 + sum CFD)

The official published CFD matrices are embedded below (mismatch_score.pkl,
240 entries; pam_scores.pkl, 16 entries - Doench 2016, as distributed by the
CRISPOR project's public CFD_Scoring files; values verified against our
documented numbers: GG=1.0, AG=0.259259, CG=0.107143, GA=0.069444,
TG=0.038961, GC=0.022222, GT=0.016129).

Query path: exhaustive vectorised candidate verification - identical
semantics to the pilot's brute-force REFERENCE path (which the seed-index
path was validated to match). chr22 queries take milliseconds, so the
seed-index acceleration layer is reserved for genome-scale shard work
(see V2B_RESEARCH/V2B_SCALE_STRESS_REPORT.md). This file does NOT claim
seed-indexing runs in-app.

Fixes vs V1 (src/specificity.py stays untouched, kept as the record):
    1. TRUE CCN detection: reverse sites require C immediately followed by C
       (V1 flagged ANY single C -> 3.48x overcount on chr22:
        9,160,652 fake vs 2,629,104 true candidate sites).
    2. Published per-mismatch CFD (position AND nucleotide-pair specific)
       replaces V1's hand-set CFD-INSPIRED position weights.
    3. Reverse-strand per-position mapping fixed (V1 weighted reverse hits
       at mirrored positions).
    4. Genomic-position dedupe for both-strand double-flags.

CFD per site = product over mismatches of MM_SCORE['rX:dY,pos']
(positions 1..20, 20 = PAM-proximal; guide 'T' is written as RNA 'U';
'dY' is the TEMPLATE-strand base = complement of the protospacer letter)
times PAM_SCORE[pam[-2:]] (canonical NGG / N-of-CCN -> 'GG' = 1.0).

HONEST SCOPE: scans only the chromosome files in data/reference/
(default chr22, hg38, ~1/23 of the genome). chr22-only != genome-wide.
Sequence-based estimate - NOT GUIDE-seq or any experimental assay.

Interface is a drop-in for src/specificity.py:
    build_index(fasta_paths, out=CACHE) / load_index(path=CACHE)
    score_guide(guide, idx) -> (specificity_0_to_1, counts_dict, n_offtarget)

Self-checks (run these first, before --build):
    py src/specificity_v2b.py --cfd-check   matrix fidelity + hand-computed CFD cases
    py src/specificity_v2b.py --selftest    planted synthetic genome vs independent
                                            pure-Python brute-force enumerator
    py src/specificity_v2b.py --build       build chr22 index (data/reference/*.fa.gz)
    py src/specificity_v2b.py --guide GAGT...  score one guide against the index
"""
import argparse, glob, gzip, os, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CACHE = 'results/reference_windows_v2b.npz'
BASE2CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
CODE2BASE = 'ACGT'
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# ---- embedded published CFD tables (Doench 2016, via CRISPOR CFD_Scoring) ----
MM_SCORE = {
    'rA:dA,1': 1.000000,
    'rA:dA,10': 0.882353,
    'rA:dA,11': 0.307692,
    'rA:dA,12': 0.333333,
    'rA:dA,13': 0.300000,
    'rA:dA,14': 0.533333,
    'rA:dA,15': 0.200000,
    'rA:dA,16': 0.000000,
    'rA:dA,17': 0.133333,
    'rA:dA,18': 0.500000,
    'rA:dA,19': 0.538462,
    'rA:dA,2': 0.727273,
    'rA:dA,20': 0.600000,
    'rA:dA,3': 0.705882,
    'rA:dA,4': 0.636364,
    'rA:dA,5': 0.363636,
    'rA:dA,6': 0.714286,
    'rA:dA,7': 0.437500,
    'rA:dA,8': 0.428571,
    'rA:dA,9': 0.600000,
    'rA:dC,1': 1.000000,
    'rA:dC,10': 0.555556,
    'rA:dC,11': 0.650000,
    'rA:dC,12': 0.722222,
    'rA:dC,13': 0.652174,
    'rA:dC,14': 0.466667,
    'rA:dC,15': 0.650000,
    'rA:dC,16': 0.192308,
    'rA:dC,17': 0.176471,
    'rA:dC,18': 0.400000,
    'rA:dC,19': 0.375000,
    'rA:dC,2': 0.800000,
    'rA:dC,20': 0.764706,
    'rA:dC,3': 0.611111,
    'rA:dC,4': 0.625000,
    'rA:dC,5': 0.720000,
    'rA:dC,6': 0.714286,
    'rA:dC,7': 0.705882,
    'rA:dC,8': 0.733333,
    'rA:dC,9': 0.666667,
    'rA:dG,1': 0.857143,
    'rA:dG,10': 0.333333,
    'rA:dG,11': 0.400000,
    'rA:dG,12': 0.263158,
    'rA:dG,13': 0.210526,
    'rA:dG,14': 0.214286,
    'rA:dG,15': 0.272727,
    'rA:dG,16': 0.000000,
    'rA:dG,17': 0.176471,
    'rA:dG,18': 0.190476,
    'rA:dG,19': 0.206897,
    'rA:dG,2': 0.785714,
    'rA:dG,20': 0.227273,
    'rA:dG,3': 0.428571,
    'rA:dG,4': 0.352941,
    'rA:dG,5': 0.500000,
    'rA:dG,6': 0.454545,
    'rA:dG,7': 0.437500,
    'rA:dG,8': 0.428571,
    'rA:dG,9': 0.571429,
    'rC:dA,1': 1.000000,
    'rC:dA,10': 0.941176,
    'rC:dA,11': 0.307692,
    'rC:dA,12': 0.538462,
    'rC:dA,13': 0.700000,
    'rC:dA,14': 0.733333,
    'rC:dA,15': 0.066667,
    'rC:dA,16': 0.307692,
    'rC:dA,17': 0.466667,
    'rC:dA,18': 0.642857,
    'rC:dA,19': 0.461538,
    'rC:dA,2': 0.909091,
    'rC:dA,20': 0.300000,
    'rC:dA,3': 0.687500,
    'rC:dA,4': 0.800000,
    'rC:dA,5': 0.636364,
    'rC:dA,6': 0.928571,
    'rC:dA,7': 0.812500,
    'rC:dA,8': 0.875000,
    'rC:dA,9': 0.875000,
    'rC:dC,1': 0.913043,
    'rC:dC,10': 0.388889,
    'rC:dC,11': 0.250000,
    'rC:dC,12': 0.444444,
    'rC:dC,13': 0.136364,
    'rC:dC,14': 0.000000,
    'rC:dC,15': 0.050000,
    'rC:dC,16': 0.153846,
    'rC:dC,17': 0.058824,
    'rC:dC,18': 0.133333,
    'rC:dC,19': 0.125000,
    'rC:dC,2': 0.695652,
    'rC:dC,20': 0.058824,
    'rC:dC,3': 0.500000,
    'rC:dC,4': 0.500000,
    'rC:dC,5': 0.600000,
    'rC:dC,6': 0.500000,
    'rC:dC,7': 0.470588,
    'rC:dC,8': 0.642857,
    'rC:dC,9': 0.619048,
    'rC:dT,1': 1.000000,
    'rC:dT,10': 0.866667,
    'rC:dT,11': 0.750000,
    'rC:dT,12': 0.714286,
    'rC:dT,13': 0.384615,
    'rC:dT,14': 0.350000,
    'rC:dT,15': 0.222222,
    'rC:dT,16': 1.000000,
    'rC:dT,17': 0.466667,
    'rC:dT,18': 0.538462,
    'rC:dT,19': 0.428571,
    'rC:dT,2': 0.727273,
    'rC:dT,20': 0.500000,
    'rC:dT,3': 0.866667,
    'rC:dT,4': 0.842105,
    'rC:dT,5': 0.571429,
    'rC:dT,6': 0.928571,
    'rC:dT,7': 0.750000,
    'rC:dT,8': 0.650000,
    'rC:dT,9': 0.857143,
    'rG:dA,1': 1.000000,
    'rG:dA,10': 0.812500,
    'rG:dA,11': 0.384615,
    'rG:dA,12': 0.384615,
    'rG:dA,13': 0.300000,
    'rG:dA,14': 0.266667,
    'rG:dA,15': 0.142857,
    'rG:dA,16': 0.000000,
    'rG:dA,17': 0.250000,
    'rG:dA,18': 0.666667,
    'rG:dA,19': 0.666667,
    'rG:dA,2': 0.636364,
    'rG:dA,20': 0.700000,
    'rG:dA,3': 0.500000,
    'rG:dA,4': 0.363636,
    'rG:dA,5': 0.300000,
    'rG:dA,6': 0.666667,
    'rG:dA,7': 0.571429,
    'rG:dA,8': 0.625000,
    'rG:dA,9': 0.533333,
    'rG:dG,1': 0.714286,
    'rG:dG,10': 0.400000,
    'rG:dG,11': 0.428571,
    'rG:dG,12': 0.529412,
    'rG:dG,13': 0.421053,
    'rG:dG,14': 0.428571,
    'rG:dG,15': 0.272727,
    'rG:dG,16': 0.000000,
    'rG:dG,17': 0.235294,
    'rG:dG,18': 0.476190,
    'rG:dG,19': 0.448276,
    'rG:dG,2': 0.692308,
    'rG:dG,20': 0.428571,
    'rG:dG,3': 0.384615,
    'rG:dG,4': 0.529412,
    'rG:dG,5': 0.785714,
    'rG:dG,6': 0.681818,
    'rG:dG,7': 0.687500,
    'rG:dG,8': 0.615385,
    'rG:dG,9': 0.538462,
    'rG:dT,1': 0.900000,
    'rG:dT,10': 0.933333,
    'rG:dT,11': 1.000000,
    'rG:dT,12': 0.933333,
    'rG:dT,13': 0.923077,
    'rG:dT,14': 0.750000,
    'rG:dT,15': 0.941176,
    'rG:dT,16': 1.000000,
    'rG:dT,17': 0.933333,
    'rG:dT,18': 0.692308,
    'rG:dT,19': 0.714286,
    'rG:dT,2': 0.846154,
    'rG:dT,20': 0.937500,
    'rG:dT,3': 0.750000,
    'rG:dT,4': 0.900000,
    'rG:dT,5': 0.866667,
    'rG:dT,6': 1.000000,
    'rG:dT,7': 1.000000,
    'rG:dT,8': 1.000000,
    'rG:dT,9': 0.642857,
    'rU:dC,1': 0.956522,
    'rU:dC,10': 0.500000,
    'rU:dC,11': 0.400000,
    'rU:dC,12': 0.500000,
    'rU:dC,13': 0.260870,
    'rU:dC,14': 0.000000,
    'rU:dC,15': 0.050000,
    'rU:dC,16': 0.346154,
    'rU:dC,17': 0.117647,
    'rU:dC,18': 0.333333,
    'rU:dC,19': 0.250000,
    'rU:dC,2': 0.840000,
    'rU:dC,20': 0.176471,
    'rU:dC,3': 0.500000,
    'rU:dC,4': 0.625000,
    'rU:dC,5': 0.640000,
    'rU:dC,6': 0.571429,
    'rU:dC,7': 0.588235,
    'rU:dC,8': 0.733333,
    'rU:dC,9': 0.619048,
    'rU:dG,1': 0.857143,
    'rU:dG,10': 0.533333,
    'rU:dG,11': 0.666667,
    'rU:dG,12': 0.947368,
    'rU:dG,13': 0.789474,
    'rU:dG,14': 0.285714,
    'rU:dG,15': 0.272727,
    'rU:dG,16': 0.666667,
    'rU:dG,17': 0.705882,
    'rU:dG,18': 0.428571,
    'rU:dG,19': 0.275862,
    'rU:dG,2': 0.857143,
    'rU:dG,20': 0.090909,
    'rU:dG,3': 0.428571,
    'rU:dG,4': 0.647059,
    'rU:dG,5': 1.000000,
    'rU:dG,6': 0.909091,
    'rU:dG,7': 0.687500,
    'rU:dG,8': 1.000000,
    'rU:dG,9': 0.923077,
    'rU:dT,1': 1.000000,
    'rU:dT,10': 0.857143,
    'rU:dT,11': 0.750000,
    'rU:dT,12': 0.800000,
    'rU:dT,13': 0.692308,
    'rU:dT,14': 0.619048,
    'rU:dT,15': 0.578947,
    'rU:dT,16': 0.909091,
    'rU:dT,17': 0.533333,
    'rU:dT,18': 0.666667,
    'rU:dT,19': 0.285714,
    'rU:dT,2': 0.846154,
    'rU:dT,20': 0.562500,
    'rU:dT,3': 0.714286,
    'rU:dT,4': 0.476190,
    'rU:dT,5': 0.500000,
    'rU:dT,6': 0.866667,
    'rU:dT,7': 0.875000,
    'rU:dT,8': 0.800000,
    'rU:dT,9': 0.928571,
}
PAM_SCORE = {
    'AA': 0.000000,
    'AC': 0.000000,
    'AG': 0.259259,
    'AT': 0.000000,
    'CA': 0.000000,
    'CC': 0.000000,
    'CG': 0.107143,
    'CT': 0.000000,
    'GA': 0.069444,
    'GC': 0.022222,
    'GG': 1.000000,
    'GT': 0.016129,
    'TA': 0.000000,
    'TC': 0.000000,
    'TG': 0.038961,
    'TT': 0.000000,
}

# ---- lookup table: MM_LUT[guide_base_code, template_base_code, pos0] -> penalty ----
def _build_lut():
    lut = np.ones((4, 4, 20), dtype=np.float64)
    for key, v in MM_SCORE.items():
        pair, pos = key.split(',')
        rb, db = pair.split(':')
        rbc = 'ACGT'.index('T' if rb == 'rU' else rb[1])   # rU -> guide T
        dbc = 'ACGT'.index(db[1])                          # template-strand base
        lut[rbc, dbc, int(pos) - 1] = v
    return lut

MM_LUT = _build_lut()
PAM_GG = float(PAM_SCORE.get('GG', 1.0))  # canonical NGG / N-of-CCN PAM factor


def parse_fasta_gz(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        lines = (l.strip() for l in f if not l.startswith('>'))
        return ''.join(lines).upper()


def encode(seq):
    arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8).copy()
    out = np.full(len(arr), 4, dtype=np.uint8)
    for i, b in enumerate(b'ACGT'):
        out[arr == b] = i
    return out


def rc_codes(g):
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[g[::-1]]


def pack_windows(rows):
    out = np.zeros(len(rows), dtype=np.uint64)
    for k in range(20):
        out |= rows[:, k].astype(np.uint64) << (2 * k)
    return out


def cfd_of_hit(guide, proto, pam):
    """Published CFD for one site. guide/proto are 20-mers in GUIDE orientation
    (proto = protospacer letters); pam = 3-letter PAM in guide orientation."""
    cfd = PAM_SCORE.get(pam[-2:], 0.0)
    for i, (gb, tb) in enumerate(zip(guide, proto)):
        if gb != tb:
            rb = 'U' if gb == 'T' else gb
            cfd *= MM_SCORE.get(f'r{rb}:d{COMP[tb]},{i + 1}', 0.0)
    return cfd


def build_index(fasta_paths, out=CACHE):
    t0 = time.time()
    gg_p, cc_p, gg_pos, cc_pos, gg_mask, cc_mask = [], [], [], [], [], []
    for path in fasta_paths:
        name = os.path.basename(path)
        arr = encode(parse_fasta_gz(path))
        n = len(arr)
        # NGG sites: i where arr[i+1]==G and arr[i+2]==G  (i = N position)
        is_g = arr == 2
        gg_i = np.where(is_g[1:-1] & is_g[2:])[0]
        gg_i = gg_i[(gg_i >= 20) & (gg_i + 2 < n)]
        # TRUE CCN sites (reverse-strand PAM): i where arr[i]==C AND arr[i+1]==C
        # (V1 bug: flagged ANY single C -> 3.48x overcount on chr22)
        is_c = arr == 1
        cc_i = np.where(is_c[:-2] & is_c[1:-1])[0]
        cc_i = cc_i[(cc_i >= 20) & (cc_i + 2 < n)]

        W = sliding_window_view(arr, 20)
        g_rows = W[gg_i - 20]
        c_rows = W[cc_i - 20]
        gg_p.append(pack_windows(g_rows))
        cc_p.append(pack_windows(c_rows))
        gg_pos.append(gg_i.astype(np.int64))
        cc_pos.append(cc_i.astype(np.int64))
        gg_mask.append((g_rows == 4).any(axis=1).astype(np.uint8))
        cc_mask.append((c_rows == 4).any(axis=1).astype(np.uint8))
        print(f"  {name}: {n:,} bp -> {len(gg_i):,} NGG sites, {len(cc_i):,} true CCN sites")
        del W, g_rows, c_rows
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(
        out,
        gg=np.concatenate(gg_p), ccn=np.concatenate(cc_p),
        gg_pos=np.concatenate(gg_pos), ccn_pos=np.concatenate(cc_pos),
        gg_mask=np.concatenate(gg_mask), ccn_mask=np.concatenate(cc_mask))
    print(f"Index saved to {out} in {time.time()-t0:.1f}s")


def load_index(path=CACHE):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in ('gg', 'ccn', 'gg_pos', 'ccn_pos', 'gg_mask', 'ccn_mask')}


def _pack_query(g):
    q = np.uint64(0)
    for k in range(20):
        q |= np.uint64(g[k]) << (2 * k)
    return q


def _hits(guide, idx):
    """All validated hits: (pos, n_mm, cfd) arrays, deduped by genomic position."""
    guide = guide.upper().replace('U', 'T')
    if len(guide) != 20 or any(c not in BASE2CODE for c in guide):
        return (np.array([], dtype=np.int64),) * 3
    g = np.array([BASE2CODE[c] for c in guide], dtype=np.uint8)
    rg = rc_codes(g)
    strands = [(idx['gg'], idx['gg_pos'], idx['gg_mask'], _pack_query(g), 'fw'),
               (idx['ccn'], idx['ccn_pos'], idx['ccn_mask'], _pack_query(rg), 'rev')]
    pos_l, mm_l, cfd_l = [], [], []
    for packed, pam_pos, mask, q, strand in strands:
        if len(packed) == 0:
            continue
        x = packed ^ q
        mism = np.bitwise_count((x | (x >> 1)) & np.uint64(0x5555555555555555))
        valid = (mism <= 3) & (mask == 0)
        if not valid.any():
            continue
        pv = packed[valid]
        m = mism[valid]
        pen = np.ones(len(m), dtype=np.float64)
        for k in range(20):
            b1 = ((pv >> np.uint64(2 * k)) & np.uint64(3)).astype(np.uint8)
            if strand == 'fw':
                # proto letter = window base; template base = complement(3-x); pos = k+1
                db, pidx, rb, cmp_b = 3 - b1, k, int(g[k]), int(g[k])
            else:
                # forward window slot k pairs with reverse-complement guide slot;
                # proto in guide orientation = complement(window); template = window
                # base itself; guide position (1-based) = 20-k
                db, pidx, rb, cmp_b = b1, 19 - k, int(g[19 - k]), int(rg[k])
            dk = b1 != np.uint8(cmp_b)
            if dk.any():
                pen[dk] *= MM_LUT[rb, db[dk], pidx]
        pos_l.append(pam_pos[valid].astype(np.int64))
        mm_l.append(m.astype(np.int64))
        cfd_l.append(pen * PAM_GG)
    if not pos_l:
        return (np.array([], dtype=np.int64),) * 3
    pos = np.concatenate(pos_l)
    mm = np.concatenate(mm_l)
    cfd = np.concatenate(cfd_l)
    _, uq = np.unique(pos, return_index=True)   # dedupe genomic positions
    return pos[uq], mm[uq], cfd[uq]


def score_guide(guide, idx):
    """Return (specificity 0-1, counts dict, n_offtarget_sites) -- same shape as V1."""
    pos, mm, cfd = _hits(guide, idx)
    counts = {'exact': int((mm == 0).sum()), '1mm': int((mm == 1).sum()),
              '2mm': int((mm == 2).sum()), '3mm': int((mm == 3).sum())}
    on_target = 1 if counts['exact'] >= 1 else 0
    extra_exact = max(0, counts['exact'] - on_target)
    # extra exact-duplicate sites elsewhere in the reference: perfect match -> CFD 1.0 each
    burden = float(cfd[mm >= 1].sum()) + extra_exact * PAM_GG
    specificity = 1.0 / (1.0 + burden)
    n_off = int((mm >= 1).sum()) + extra_exact
    return specificity, counts, n_off


# ============ self-verification (independent of the numpy fast path) ============

def _brute_force_hits(guide, seq):
    """Independent pure-Python reference: enumerate TRUE NGG + true CCn sites,
    keep <=3 mismatches, dedupe by position. Returns {pos: (n_mm, cfd)}."""
    guide = guide.upper()
    n = len(seq)
    hits = {}

    def rc(s):
        return ''.join(COMP[c] for c in reversed(s))

    for i in range(20, n - 2):
        if seq[i + 1] == 'G' and seq[i + 2] == 'G':          # forward NGG
            proto = seq[i - 20:i]
            if 'N' not in proto:
                mm = sum(a != b for a, b in zip(guide, proto))
                if mm <= 3:
                    hits[i] = (mm, cfd_of_hit(guide, proto, 'AGG'))
        if seq[i] == 'C' and seq[i + 1] == 'C':              # reverse TRUE CCn
            raw = seq[i - 20:i]
            if 'N' not in raw:
                proto = rc(raw)
                mm = sum(a != b for a, b in zip(guide, proto))
                if mm <= 3:
                    hits[i] = (mm, cfd_of_hit(guide, proto, 'AGG'))
    return hits


def _selftest():
    rng = np.random.default_rng(7)
    seq = list(''.join(rng.choice(list('ACGT'), 6000)))
    guide = 'GAGTCCGAGCAGAAGAAGAA'

    def rc(s):
        return ''.join(COMP[c] for c in reversed(s))

    def plant_fwd(start, proto, pam='AGG'):
        seq[start:start + 20] = list(proto)
        seq[start + 20:start + 23] = list(pam)
        return start + 20          # N position of the PAM

    def plant_rev(pam_start, proto_guide_oriented, pam_fwd='CCA'):
        seq[pam_start - 20:pam_start] = list(rc(proto_guide_oriented))
        seq[pam_start:pam_start + 3] = list(pam_fwd)
        return pam_start           # forward position of the C of CCN

    def mutate(proto, edits):
        p = list(proto)
        for pos1, b in edits:
            p[pos1 - 1] = b
        return ''.join(p)

    planted = {}   # pos -> (case, expect_hit)
    G = guide
    # 1. exact copy, forward                                    -> hit, mm 0
    planted[plant_fwd(200, G)] = ('fwd exact', True)
    # 2. one PAM-proximal mismatch (pos 17), forward            -> hit, mm 1
    planted[plant_fwd(400, mutate(G, [(17, 'T' if G[16] != 'T' else 'A')]))] = ('fwd 1mm@17', True)
    # 3. two mismatches, forward                                -> hit, mm 2
    alt = lambda b: 'A' if b != 'A' else 'C'
    planted[plant_fwd(600, mutate(G, [(3, alt(G[2])), (19, alt(G[18]))]))] = ('fwd 2mm', True)
    # 4. three mismatches, forward                              -> hit, mm 3
    planted[plant_fwd(800, mutate(G, [(1, alt(G[0])), (10, alt(G[9])), (20, alt(G[19]))]))] = ('fwd 3mm', True)
    # 5. four mismatches, forward                               -> DECOY, no hit
    planted[plant_fwd(1000, mutate(G, [(1, alt(G[0])), (5, alt(G[4])), (10, alt(G[9])), (20, alt(G[19]))]))] = ('fwd 4mm decoy', False)
    # 6. exact copy on REVERSE strand with true CCn PAM         -> hit, mm 0
    planted[plant_rev(1220, G)] = ('rev exact (true CCn)', True)
    # 7. reverse strand, one mismatch                           -> hit, mm 1
    planted[plant_rev(1420, mutate(G, [(7, alt(G[6]))]))] = ('rev 1mm', True)
    # 8. V1 CCN-BUG REGRESSION: single C, next base NOT C, but
    #    upstream 20-mer is a <=3mm match on the reverse strand -> DECOY, no hit
    dpos = 1620
    seq[dpos - 20:dpos] = list(rc(mutate(G, [(20, alt(G[19]))])))
    seq[dpos] = 'C'; seq[dpos + 1] = 'A'; seq[dpos + 2] = 'T'
    planted[dpos] = ('single-C decoy (V1 would count!)', False)
    # 9. exact copy but an N inside the protospacer window      -> DECOY (masked)
    nseq = mutate(G, [])
    npos = plant_fwd(1800, nseq)
    seq[npos - 10] = 'N'
    planted[npos] = ('N-in-window decoy', False)

    fwd = ''.join(seq)
    hits = _brute_force_hits(G, fwd)
    print(f"selftest: brute-force reference found {len(hits)} hit positions for guide {G}")

    exp_pos = sorted(p for p, (_, keep) in planted.items() if keep)
    got_pos = sorted(hits)
    assert got_pos == exp_pos, f"hit-position mismatch!\n got: {got_pos}\n exp: {exp_pos}"
    print("  positions match planted expectation exactly (incl. all 3 decoys rejected)")

    # mm tiers from the reference
    tiers = {0: 0, 1: 0, 2: 0, 3: 0}
    for p, (m, cf) in hits.items():
        tiers[m] += 1
    print(f"  reference tiers: exact={tiers[0]} 1mm={tiers[1]} 2mm={tiers[2]} 3mm={tiers[3]}")

    # run the FAST path on the same sequence through build_index machinery
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.fa', delete=False) as tf:
        tf.write('>synth\n' + fwd + '\n')
        tmpfa = tf.name
    tmpidx = tmpfa + '.npz'
    build_index([tmpfa], out=tmpidx)
    idx = load_index(tmpidx)
    pos, mm, cfd = _hits(G, idx)
    assert sorted(pos.tolist()) == exp_pos, f"fast-path positions mismatch: {sorted(pos.tolist())}"
    ref_by_pos = {p: (m, cf) for p, (m, cf) in hits.items()}
    for p, m, cf in zip(pos.tolist(), mm.tolist(), cfd.tolist()):
        em, ecf = ref_by_pos[p]
        assert m == em, f"pos {p}: mm {m} != {em}"
        assert abs(cf - ecf) < 1e-9, f"pos {p}: cfd {cf} != {ecf}"
    print("  fast path == reference on EVERY hit (positions, mm, CFD to 1e-9)")

    spec, counts, n_off = score_guide(G, idx)
    print(f"  score_guide -> specificity={spec:.4f} counts={counts} n_off={n_off}")
    exp_counts = {'exact': tiers[0], '1mm': tiers[1], '2mm': tiers[2], '3mm': tiers[3]}
    assert counts == exp_counts, f"{counts} != {exp_counts}"
    print("SELFTEST: PASS")


def _cfd_check():
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("matrix fidelity:")
    chk("240 mismatch entries", len(MM_SCORE) == 240)
    chk("16 PAM entries", len(PAM_SCORE) == 16)
    chk("positions 1..20 complete", all(f"rA:dC,{p}" in MM_SCORE for p in range(1, 21)))
    chk("GG = 1.0", PAM_SCORE['GG'] == 1.0)
    for dinuc, v in [('AG', 0.259259), ('CG', 0.107143), ('GA', 0.069444),
                     ('TG', 0.038961), ('GC', 0.022222), ('GT', 0.016129)]:
        chk(f"{dinuc} = {v}", abs(PAM_SCORE[dinuc] - v) < 1e-4)

    print("hand-computed CFD cases:")
    near = lambda a, b: abs(a - b) < 1e-6
    chk("perfect match = 1.0", cfd_of_hit('A' * 20, 'A' * 20, 'AGG') == 1.0)
    # single mismatch rU:dT at pos 12 (guide T vs proto A) -> 0.8 (published value)
    g = 'A' * 11 + 'T' + 'A' * 8
    chk("single rU:dT,12 = 0.8", near(cfd_of_hit(g, 'A' * 20, 'AGG'), MM_SCORE['rU:dT,12']))
    chk("... value is 0.8", near(MM_SCORE['rU:dT,12'], 0.8))
    # single mismatch rU:dC at pos 5 (guide T vs proto G) -> 0.64 (published value)
    g2 = 'T' * 20
    p2 = 'T' * 4 + 'G' + 'T' * 15
    chk("single rU:dC,5 = 0.64", near(cfd_of_hit(g2, p2, 'AGG'), 0.64))
    # two mismatches multiply
    p3 = 'A' * 2 + 'C' + 'A' * 20  # proto pos3 C, guide A -> rA:dG,3
    two = cfd_of_hit(g, p3[:23] and p3[3:23] if False else ('A' * 2 + 'C' + 'A' * 17), 'AGG')
    chk("two mismatches multiply", near(two, MM_SCORE['rU:dT,12'] * MM_SCORE['rA:dG,3']))
    # alternative PAM NAG multiplies by 0.259259
    chk("NAG PAM factor = 0.259259", near(cfd_of_hit('GAGTCCGAGCAGAAGAAGAA', 'GAGTCCGAGCAGAAGAAGAA', 'AAG'), 0.259259))
    print("CFD-CHECK: " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--guide', default=None)
    ap.add_argument('--cfd-check', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--refs', default='data/reference/*.fa.gz')
    args = ap.parse_args()
    if args.cfd_check:
        sys.exit(0 if _cfd_check() else 1)
    elif args.selftest:
        _selftest()
    elif args.build:
        paths = sorted(glob.glob(args.refs))
        if not paths:
            print(f"No reference files found matching {args.refs}")
            print("Download chr22 (hg38): https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz")
            print("and put it in data/reference/")
        else:
            build_index(paths)
    elif args.guide:
        idx = load_index()
        if idx is None:
            print(f"Index not found ({CACHE}). Run: py src/specificity_v2b.py --build")
        else:
            t0 = time.time()
            spec, counts, n_off = score_guide(args.guide, idx)
            print(f"Guide: {args.guide.upper()}")
            print(f"On-target sites: {counts['exact']} | 1mm: {counts['1mm']} | 2mm: {counts['2mm']} | 3mm: {counts['3mm']}")
            print(f"Off-target sites: {n_off} | Specificity: {spec:.4f} | scored in {time.time()-t0:.2f}s")
    else:
        print(__doc__)
