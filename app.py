"""
CRISPR Multi-Objective Guide Designer — polished Streamlit presentation layer.
Backend: existing V1 application (unchanged algorithms).
  - Activity : CNN trained on DeepHF 55,604 real guides (V1 test Spearman 0.745;
               RF 0.728 on the same random split). Model: results/cnn_baseline.pt
  - Structure: real ViennaRNA MFE
  - Specificity: V1 sequence-based CFD-INSPIRED scan vs human chr22 (hg38),
                 both strands, NGG PAM — chr22 only, NOT genome-wide, NOT published
                 CFD, NOT GUIDE-seq. (V2B not integrated.)
  - Ranking  : Pareto-optimal set over [efficiency, specificity, structure].
No weighted composite score.
"""
import re, os
import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.specificity import score_guide
    HAS_SPEC_MOD = True
except Exception:
    HAS_SPEC_MOD = False

# ---------- model (unchanged V1) ----------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.fc(self.conv(x))


def seq_to_4x20(seqs):
    arr = np.zeros((len(seqs), 4, 20))
    for idx, s in enumerate(seqs):
        for i, ch in enumerate(s[:20]):
            if ch in BASES:
                arr[idx, BASES[ch], i] = 1
    return arr


@st.cache_resource
def load_cnn():
    p = 'results/cnn_baseline.pt'
    if not HAS_TORCH or not os.path.exists(p):
        return None
    try:
        m = SimpleCNN()
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.eval()
        return m
    except Exception:
        return None


cnn = load_cnn()

try:
    import RNA
    HAS_VIENNA = True
except Exception:
    HAS_VIENNA = False

# ---------- page setup ----------
st.set_page_config(page_title="CRISPR Multi-Objective Guide Designer", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# PRESENTATION LAYER — dark biotech theme (fonts, background, components)
# =====================================================================



# ---------- HERO BANNER ----------
st.title("🧬 CRISPR Multi-Objective Guide Designer")
st.caption("CNN efficiency · real ViennaRNA structure · sequence-based specificity vs chr22 · Pareto-optimal multi-objective ranking")



if cnn is None:
    st.error("results/cnn_baseline.pt not found (or torch missing) — run the DeepHF training first.")
    st.stop()

st.success("✅ CNN loaded — DeepHF 55,604 real guides · V1 test Spearman **0.745** (RF 0.728 on the same random split)")

# ---------- real specificity index (V1 chr22 backend — unchanged) ----------
@st.cache_resource
def load_spec_index():
    try:
        from src.specificity import load_index
        return load_index()
    except Exception:
        return None


spec_idx = load_spec_index()
if spec_idx is not None:
    st.success("✅ Specificity: sequence-based CFD-inspired scan vs human chr22 (hg38), both strands, NGG PAM")
else:
    st.warning("⚠️ Reference index not found — using the fallback sequence proxy for Specificity.\n"
               "Run once: `py src\\specificity.py --build` (needs `data/reference/chr22.fa.gz`, see Method).")

# ---------- objective helpers (unchanged V1) ----------
OBJ = ["Efficiency_CNN", "Specificity", "Structure"]


def gc_pct(seq):
    return (seq.count('G') + seq.count('C')) / len(seq) * 100


def gc_score(seq):
    gc = (seq.count('G') + seq.count('C')) / len(seq)
    return 1.0 if 0.4 <= gc <= 0.65 else max(0.0, 1 - 2 * abs(gc - 0.52))


def mfe_score(seq):
    if HAS_VIENNA:
        ss, mfe = RNA.fold(seq)
        s = 1.0 if mfe > -4 else max(0.0, 1 - (abs(mfe) - 4) / 8)
        return mfe, s, ss
    return -len(seq) * 0.15, 0.7, '.' * len(seq)


def specificity_proxy(seq):
    """PRELIMINARY deterministic specificity proxy (fallback when index missing)."""
    penalty = 0.0
    if len(set(seq[-8:])) <= 2:
        penalty += 0.25
    gc = (seq.count('G') + seq.count('C')) / len(seq)
    if gc > 0.75:
        penalty += 0.15
    elif gc < 0.25:
        penalty += 0.10
    for run in ('AAAA', 'CCCC', 'GGGG', 'TTTT'):
        if run in seq:
            penalty += 0.05
    return float(np.clip(1.0 - penalty, 0.05, 1.0))


def is_pareto_efficient(costs, maximize=True):
    if maximize:
        costs = -costs
    n = costs.shape[0]
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if is_eff[i]:
            is_eff[is_eff] = np.any(costs[is_eff] < costs[i], axis=1)
            is_eff[i] = True
    return is_eff


def dominates(a, b):
    return all(a[o] >= b[o] for o in OBJ) and any(a[o] > b[o] for o in OBJ)


def why_pareto(row, df):
    if row["Pareto"]:
        return ("**Pareto-optimal** — no other candidate guide is better on ALL objectives "
                "(efficiency, specificity, structure). Improving any one objective would force "
                "a trade-off in another. That's the point: *you* choose the trade-off.")
    dominators = df[df.apply(lambda r: dominates(r, row), axis=1)]
    if len(dominators):
        d = dominators.sort_values("Efficiency_CNN", ascending=False).iloc[0]
        return (f"**Dominated by `{d['Guide']}`** — it is at least as good on everything "
                f"and strictly better on at least one objective "
                f"(Eff {d['Efficiency_CNN']:.3f} vs {row['Efficiency_CNN']:.3f}, "
                f"Spec {d['Specificity']:.3f} vs {row['Specificity']:.3f}).")
    return "Not Pareto-optimal (another guide is strictly better on at least one objective, equal on the rest)."


# ---------- sidebar controls ----------
st.sidebar.markdown("### 🎛 Controls")
sort_by = st.sidebar.selectbox("Sort table by", OBJ + ["GC%"], index=0)
min_eff = st.sidebar.slider("Minimum efficiency", 0.0, 1.0, 0.0, 0.05)
only_pareto = st.sidebar.checkbox("Show only Pareto-optimal", value=False)

st.sidebar.markdown("### 📈 Plot axes")
x_axis = st.sidebar.selectbox("X axis", ["Specificity", "Efficiency_CNN", "Structure", "GC%"], index=0)
y_axis = st.sidebar.selectbox("Y axis", ["Efficiency_CNN", "Specificity", "Structure", "GC%"], index=1)

# ---------- REAL GENE PRESETS (from NCBI RefSeq — existing) ----------
PRESETS = {
    "HBB (beta-globin) CDS start — NM_000518.5": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGAT",
    "TP53 (tumor suppressor p53) CDS start — NM_000546.6": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAG",
    "BRCA1 (breast cancer 1) CDS start — NM_007294.4": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGACACAGGTTTGGAG",
    "CFTR (cystic fibrosis transmembrane conductance regulator) CDS start — NM_000492.4": "ATGCAGAGGTCGCCTCTGGAAAAGGCCAGCGTTGTCTCCAAACTTTTTTTCAGCTGGACCAGACCAATTTTGAGGAAAGGATACAGACAGCGCCTGGAATTGTCAGACATATACCAAATCCCTTCTGTTGATTCTGCTGACAATCTATCTGAAAAATTGGAAAGAGAATGGGATAGAGAGCTGGCTTCAAAGAAAAATCCTAAACTCATTAATGCCCTTCGGCGATGTTTTTTCTGGAGATTTATGTTCTATGGAATCTTTTTATATTTAGGGGAAGTCACCAAAGCAGTACAGCCTCTC",
    "BRCA2 (breast cancer 2, DNA repair) CDS start — NM_000059.4": "ATGCCTATTGGATCCAAAGAGAGGCCAACATTTTTTGAAATTTTTAAGACACGCTGCAACAAAGCAGATTTAGGACCAATAAGTCTTAATTGGTTTGAAGAACTTTCTTCAGAAGCTCCACCCTATAATTCTGAACCTGCAGAAGAATCTGAACATAAAAACAACAATTACGAACCAAACCTATTTAAAACTCCACAAAGGAAACCATCTTATAATCAGCTGGCTTCAACTCCAATAATATTCAAAGAGCAAGGGCTGACTCTGCCGCTGTACCAATCTCCTGTAAAAGAATTAGATAAA",
    "EGFR (epidermal growth factor receptor) CDS start — NM_005228.5": "ATGCGACCCTCCGGGACGGCCGGGGCAGCGCTCCTGGCGCTGCTGGCTGCGCTCTGCCCGGCGAGTCGGGCTCTGGAGGAAAAGAAAGTTTGCCAAGGCACGAGTAACAAGCTCACGCAGTTGGGCACTTTTGAAGATCATTTTCTCAGCCTCCAGAGGATGTTCAATAACTGTGAGGTGGTCCTTGGGAATTTGGAAATTACCTATGTGCAGAGGAATTATGATCTTTCCTTCTTAAAGACCATCCAGGAGGTGGCTGGTTATGTCCTCATTGCCCTCAACACAGTGGAGCGAATTCCT",
    "KRAS (KRAS proto-oncogene GTPase) CDS start — NM_004985.5": "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTAATTGATGGAGAAACCTGTCTCTTGGATATTCTCGACACAGCAGGTCAAGAGGAGTACAGTGCAATGAGGGACCAGTACATGAGGACTGGGGAGGGCTTTCTTTGTGTATTTGCCATAAATAATACTAAATCATTTGAAGATATTCACCATTATAGAGAACAAATT",
    "HTT (huntingtin) CDS start — NM_002111.8": "ATGGCGACCCTGGAAAAGCTGATGAAGGCCTTCGAGTCCCTCAAGTCCTTCCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAACAGCCGCCACCGCCGCCGCCGCCGCCGCCGCCTCCTCAGCTTCCTCAGCCGCCGCCGCAGGCACAGCCGCTGCTGCCTCAGCCGCAGCCGCCCCCGCCGCCGCCCCCGCCGCCACCCGGCCCGGCTGTGGCTGAGGAGCCGCTGCACCGACCAAAGAAAGAACTTTCAGCTACCAAGAAAGAC",
    "MYC (MYC proto-oncogene) CDS start — NM_002467.6": "ATGCCCCTCAACGTTAGCTTCACCAACAGGAACTATGACCTCGACTACGACTCGGTGCAGCCGTATTTCTACTGCGACGAGGAGGAGAACTTCTACCAGCAGCAGCAGCAGAGCGAGCTGCAGCCCCCGGCGCCCAGCGAGGATATCTGGAAGAAATTCGAGCTGCTGCCCACCCCGCCCCTGTCCCCTAGCCGCCGCTCCGGGCTCTGCTCGCCCTCCTACGTTGCGGTCACACCCTTCTCCCTTCGGGGAGACAACGACGGCGGTGGCGGGAGCTTCTCCACGGCCGACCAGCTGGAG",
    "GAPDH (glyceraldehyde-3-phosphate dehydrogenase (housekeeping)) CDS start — NM_002046.7": "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGCAAATTCCATGGCACCGTCAAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTCGTGGAGTCCACTGGC",
}
st.sidebar.markdown("### 🧬 Real gene presets")
preset = st.sidebar.selectbox("Load a real gene region (NCBI RefSeq)", list(PRESETS.keys()) + ["Custom paste..."], index=0)
if st.sidebar.button("Load into input box"):
    if preset in PRESETS:
        st.session_state["seq_input"] = PRESETS[preset]
    else:
        st.session_state["seq_input"] = ""

# ---------- input ----------
seq_input = st.text_area(
    "Paste target DNA (will find all 20 bp + NGG guides):",
    height=130,
    key="seq_input",
    placeholder="GAGTCCGAGCAGAAGAAGAA GGGCTCCCATCACATCAACCGG TGGAGGAGAGAGACAGAG".replace(" ", "") +
                "  — or pick a real gene preset from the sidebar and click 'Load into input box'")

pam_pattern = re.compile(r'(?=(.{20}([ACGT])GG))')

# =====================================================================
# STATE PERSISTENCE (Streamlit rerun-safe)
# Results are computed once on "Design Guides" click and stored in
# st.session_state["guides_df"]; every rerun (inspector dropdown change,
# sidebar sort/filter/plot-axis change) re-renders from the stored frame.
# =====================================================================
def render_results():
    """Render metrics, table, plot, inspector, download from persisted results."""
    df = st.session_state["guides_df"]

    # sidebar controls apply to the PERSISTED results
    view = df[df["Efficiency_CNN"] >= min_eff].copy()
    if only_pareto:
        view = view[view["Pareto"]].copy()
    view = view.sort_values(sort_by, ascending=False).reset_index(drop=True)
    view.insert(0, "Rank", range(1, len(view) + 1))

    n_pareto = int(df["Pareto"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Guides found", len(df))
    m2.metric("Pareto-optimal", n_pareto)
    m3.metric("Best efficiency", f"{df['Efficiency_CNN'].max():.3f}")
    m4.metric("Best specificity", f"{df['Specificity'].max():.3f}")

    st.caption(f"Showing {len(view)}/{len(df)} guides" + (" (Pareto only)" if only_pareto else ""))

    show = view.copy()
    show["Pareto"] = show["Pareto"].map({True: "⭐ Pareto-optimal", False: ""})
    st.dataframe(
        show.drop(columns=["fold"]), width="stretch", hide_index=True,
        column_config={
            "Efficiency_CNN": st.column_config.ProgressColumn("Efficiency", min_value=0.0, max_value=1.0, format="%.3f"),
            "Specificity": st.column_config.ProgressColumn("Specificity", min_value=0.0, max_value=1.0, format="%.3f"),
            "Off-targets": st.column_config.TextColumn("Off-targets 1/2/3mm"),
            "Structure": st.column_config.ProgressColumn("Structure", min_value=0.0, max_value=1.0, format="%.2f"),
            "GC%": st.column_config.NumberColumn("GC%", format="%.1f"),
            "Pareto": st.column_config.TextColumn("Pareto"),
        })

    try:
        import plotly.express as px
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color="Structure",
            color_continuous_scale="RdYlGn",
            hover_data=["Guide", "PAM", "GC%", "MFE", "Efficiency_CNN", "Specificity", "Off-targets"],
            title=f"{y_axis} vs {x_axis} (color = structure; red star = Pareto-optimal)")
        p = df[df["Pareto"]]
        fig.add_scatter(
            x=p[x_axis], y=p[y_axis], mode="markers",
            marker=dict(symbol="star", size=18, color="red", line=dict(color="black", width=1)),
            name="Pareto-optimal")
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(10,18,40,0.6)",
                          font=dict(color="#e6edf7"))
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.scatter_chart(df, x=x_axis, y=y_axis, color="Structure")

    st.markdown("### 🔍 Inspect a guide")
    opts = ["⭐ " + r["Guide"] + f"  (Eff {r['Efficiency_CNN']:.3f}, Spec {r['Specificity']:.3f})"
            if r["Pareto"] else r["Guide"] + f"  (Eff {r['Efficiency_CNN']:.3f}, Spec {r['Specificity']:.3f})"
            for _, r in df.iterrows()]
    # persist the options list too so the dropdown is stable across reruns
    st.session_state["inspector_opts"] = opts
    pick = st.selectbox("Choose a guide to inspect", opts, index=0)
    sel = df.iloc[opts.index(pick)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Efficiency (CNN)", f"{sel['Efficiency_CNN']:.3f}")
    c2.metric("Specificity", f"{sel['Specificity']:.3f}")
    c3.metric("Structure", f"{sel['Structure']:.2f}")
    c4.metric("MFE (kcal/mol)", f"{sel['MFE']:.2f}")
    st.markdown(
        f"**Guide:** `{sel['Guide']}`  ·  **PAM:** `{sel['PAM']}`  ·  **GC:** {sel['GC%']:.1f}%  ·  "
        f"**Off-targets (1/2/3mm, chr22):** `{sel['Off-targets']}`  ·  **Predicted RNA fold:** `{sel['fold']}`")
    st.info(why_pareto(sel, df))

    st.download_button("⬇ Download CSV", show.to_csv(index=False), "guides_pareto.csv", "text/csv")


if st.button("🎯 Design Guides", type="primary"):
    dna = re.sub(r'[^ACGTacgt]', '', seq_input.upper())
    if len(dna) < 30:
        st.error("Please paste at least 30 bp.")
        st.session_state.pop("guides_df", None)
    else:
        guides = []
        for m in pam_pattern.finditer(dna):
            g = m.group(1)[:20]
            if len(g) == 20:
                mfe, struct, ss = mfe_score(g)
                with torch.no_grad():
                    x = torch.tensor(seq_to_4x20([g]), dtype=torch.float32)
                    eff = float(np.clip(cnn(x).numpy()[0, 0], 0, 1))
                if spec_idx is not None:
                    spec, counts, n_off = score_guide(g, spec_idx)
                    spec_val = round(spec, 3)
                    off_str = f"{counts['1mm']}/{counts['2mm']}/{counts['3mm']}"
                else:
                    spec_val = round(specificity_proxy(g), 3)
                    off_str = "n/a"
                guides.append({
                    "Guide": g,
                    "PAM": m.group(2) + "GG",
                    "GC%": round(gc_pct(g), 1),
                    "MFE": round(mfe, 2),
                    "Structure": round(struct, 2),
                    "Specificity": spec_val,
                    "Off-targets": off_str,
                    "Efficiency_CNN": round(eff, 3),
                    "fold": ss,
                })
        if not guides:
            st.warning("No NGG found. Try a different sequence (needs ...NGG).")
            st.session_state.pop("guides_df", None)
        else:
            df = pd.DataFrame(guides).drop_duplicates(subset="Guide")
            costs = np.vstack([df[o].values for o in OBJ]).T
            df["Pareto"] = is_pareto_efficient(costs, maximize=True)
            # PERSIST results so reruns (inspector/sidebar) reuse them
            st.session_state["guides_df"] = df

# -------- render persisted results on every rerun (outside the button) --------
if "guides_df" in st.session_state:
    render_results()
else:
    st.info("👆 Pick a **real gene preset** from the sidebar (HBB / TP53 / BRCA1 / CFTR / BRCA2 / EGFR / KRAS / HTT / MYC / GAPDH — real NCBI sequences) and click **Load into input box**, or paste your own DNA. Then click **Design Guides**.\n\n"
            "Tip: guides are found wherever a 20 bp stretch is followed by NGG (the Cas9 PAM).")

with st.expander("📖 Method (honest summary)"):
    st.markdown("""
**Activity prediction**

- CNN trained using **55,604 real DeepHF guides** (WT-SpCas9 indel rates).
- V1 random-split test Spearman = **0.745**
- RF baseline = **0.728** (same random split, same features: one-hot 20-mer + GC).

**Generalization validation**

- A separate V2A experiment used a **gene-level split** (all guides of test genes held out).
- CNN = **0.720**, RF = **0.718**; test = 10,944 guides from 4,034 completely held-out genes.
- The small drop **supports cross-gene generalization** — this is a separate evaluation experiment from the deployed V1 model.

**Structure**

- Real **ViennaRNA**-based RNA secondary-structure prediction (MFE ≈ 0 = unfolded = good).

**Specificity**

- The deployed interface currently uses the **V1 sequence-based chr22 scan**.
- It searches **both strands** for PAM-compatible sites and reports **1/2/3-mismatch off-target counts**.
- Its weighting is **CFD-inspired** — it is **not** the published CFD matrix.
- It is **chr22-only**, not genome-wide.
- It is a **sequence-based estimate**, not GUIDE-seq or any experimental off-target assay.

**Ranking**

- Pareto-optimal ranking is performed over the existing **efficiency / specificity / structure** objectives.
- No weighted composite score is used — you choose the trade-off.
""")
