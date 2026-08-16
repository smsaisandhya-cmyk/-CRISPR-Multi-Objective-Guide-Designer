"""
specificity.py - real CFD-inspired off-target scan against a reference genome.

Builds a compact index of PAM-flanked 20-mers from reference FASTA files
(each 20-mer packed into one 64-bit integer: 2 bits per base; mismatches are
counted with XOR + popcount), then scores each candidate guide by counting
near-match sites (<=3 mismatches + PAM) with position-weighted penalties
(seed / PAM-proximal mismatches weigh more, mirroring Doench CFD logic).

Both strands are scanned: NGG PAM sites on the forward strand (query = guide)
and CCN PAM sites, i.e. reverse-strand NGG, (query = reverse-complement of
guide). Hits are deduplicated by genomic position.

HONEST SCOPE: scans only the chromosome files you put in data/reference/
(default: chr22, hg38, ~1/23 of the human genome). Extend by adding more
chromosomes (chr11, chr17...) and re-building. Not a GUIDE-seq-level
experimental measure - it is a sequence-based CFD-style estimate.

Usage:
  py src/specificity.py --build                # build index from data/reference/*.fa.gz
  py src/specificity.py --guide GTAG...        # score one guide (uses index)
"""
import argparse, gzip, glob, os, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CACHE = 'results/reference_windows.npz'
BASE2CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# CFD-inspired position weights (PAM-distal -> PAM-proximal; seed = positions 16-19)
POS_WEIGHTS = np.array([0.14] * 8 + [0.35] * 8 + [1.0] * 4, dtype=np.float32)


def parse_fasta_gz(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        lines = (l.strip() for l in f if not l.startswith('>'))
        return ''.join(lines).upper()


def encode(seq):
    """ACGT -> 0..3, anything else (N) -> 4."""
    arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8).copy()
    out = np.full(len(arr), 4, dtype=np.uint8)
    for i, b in enumerate(b'ACGT'):
        out[arr == b] = i
    return out


def rc_codes(g):
    """Reverse complement as codes (A<->T, C<->G)."""
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[g[::-1]]


def pack_windows(rows):
    """Pack (m, 20) uint8 codes into (m,) uint64 (2 bits per base)."""
    out = np.zeros(len(rows), dtype=np.uint64)
    for k in range(20):
        out |= rows[:, k].astype(np.uint64) << (2 * k)
    return out


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
        # CCN sites (reverse-strand PAM): i where arr[i]==C
        is_c = arr == 1
        cc_i = np.where(is_c[:-2])[0]
        cc_i = cc_i[(cc_i >= 20) & (cc_i + 2 < n)]

        W = sliding_window_view(arr, 20)  # view, no copy
        g_rows = W[gg_i - 20]
        c_rows = W[cc_i - 20]
        gg_p.append(pack_windows(g_rows))
        cc_p.append(pack_windows(c_rows))
        gg_pos.append(gg_i.astype(np.int64))
        cc_pos.append(cc_i.astype(np.int64))
        gg_mask.append((g_rows == 4).any(axis=1).astype(np.uint8))
        cc_mask.append((c_rows == 4).any(axis=1).astype(np.uint8))
        print(f"  {name}: {n:,} bp -> {len(gg_i):,} NGG sites, {len(cc_i):,} CCN sites")
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


def score_guide(guide, idx):
    """Return (specificity 0-1, counts dict, n_offtarget_sites)."""
    guide = guide.upper().replace('U', 'T')
    if len(guide) != 20 or any(c not in BASE2CODE for c in guide):
        return 1.0, {'exact': 0, '1mm': 0, '2mm': 0, '3mm': 0}, 0
    g = np.array([BASE2CODE[c] for c in guide], dtype=np.uint8)
    rg = rc_codes(g)
    q_fw = _pack_query(g)
    q_rc = _pack_query(rg)

    pos_list, mm_list, w_list = [], [], []
    for packed, pam_pos, mask, q in [
            (idx['gg'], idx['gg_pos'], idx['gg_mask'], q_fw),
            (idx['ccn'], idx['ccn_pos'], idx['ccn_mask'], q_rc)]:
        if len(packed) == 0:
            continue
        # XOR gives set bits per 2-bit base pair; a base mismatch can set 1 or 2 bits,
        # so OR the two bit-planes and mask the low bit of each pair first.
        x = packed ^ q
        mism = np.bitwise_count((x | (x >> 1)) & np.uint64(0x5555555555555555))
        valid = (mism <= 3) & (mask == 0)
        if not valid.any():
            continue
        m = mism[valid]
        pos_list.append(pam_pos[valid].astype(np.int64))
        mm_list.append(m.astype(np.int64))
        # per-site penalty: sum of POS_WEIGHTS at mismatched positions
        diffs = np.zeros((m.shape[0], 20), dtype=bool)
        for k in range(20):
            b1 = (packed[valid] >> (2 * k)) & 3
            diffs[:, k] = b1 != np.uint64(g[k])
        wpos = np.where(diffs, POS_WEIGHTS, 0.0).sum(axis=1)
        w_list.append(wpos)

    if not pos_list:
        return 1.0, {'exact': 0, '1mm': 0, '2mm': 0, '3mm': 0}, 0
    pos = np.concatenate(pos_list)
    mm = np.concatenate(mm_list)
    w = np.concatenate(w_list)
    # dedupe by genomic position (both strands can flag the same site)
    _, uniq = np.unique(np.stack([pos, mm], axis=1), axis=0, return_index=True)
    pos, mm, w = pos[uniq], mm[uniq], w[uniq]

    counts = {'exact': int((mm == 0).sum()),
              '1mm': int((mm == 1).sum()),
              '2mm': int((mm == 2).sum()),
              '3mm': int((mm == 3).sum())}
    on_target = 1 if counts['exact'] >= 1 else 0
    extra_exact = max(0, counts['exact'] - on_target)
    total_w = float(w[mm >= 1].sum()) + extra_exact * 3.0
    specificity = 1.0 / (1.0 + total_w)
    n_off = int((mm >= 1).sum()) + extra_exact
    return specificity, counts, n_off


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--guide', default=None)
    ap.add_argument('--refs', default='data/reference/*.fa.gz')
    args = ap.parse_args()
    if args.build:
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
            print(f"Index not found ({CACHE}). Run: py src/specificity.py --build")
        else:
            t0 = time.time()
            spec, counts, n_off = score_guide(args.guide, idx)
            print(f"Guide: {args.guide.upper()}")
            print(f"On-target sites: {counts['exact']} | 1mm: {counts['1mm']} | 2mm: {counts['2mm']} | 3mm: {counts['3mm']}")
            print(f"Off-target sites: {n_off} | Specificity: {spec:.4f} | scored in {time.time()-t0:.2f}s")
