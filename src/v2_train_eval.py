"""
v2_train_eval.py - V2A: gene-level split training/evaluation.

The ONLY change vs V1 (evaluate.py) is the SPLIT:
  V1: random 80/10/10 split on GUIDES
  V2A: split on GENES (80% of genes train, 20% test); all guides of a
       test gene held out; train genes further split 80/20 into train/val.

Everything else is IDENTICAL to V1:
  - features: one-hot(4x20) + GC
  - RF: n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
  - CNN: 2x Conv1d(4->32->64, k=3) + AvgPool + FC(64->32->1) + Sigmoid
  - Adam lr=1e-3, MSE, batch 256, 15 epochs, best-val epoch restored
  - metrics: Spearman + RMSE on held-out TEST set

Input : results/v2/gene_labels.csv (mapping output: Sequence,PAM,Activity,Gene_Symbol,Entrez_ID)
Output: results/v2/v2a_metrics.csv  (append-style, V2 namespace)
        results/v2/v2a_genes5plus_metrics.csv (secondary: genes with >=5 guides)
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}


def gc_content(seq):
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if len(seq) else 0.0


def one_hot_4x20(seqs):
    arr = np.zeros((len(seqs), 4, 20))
    for idx, s in enumerate(seqs):
        for i, ch in enumerate(s[:20]):
            if ch in BASES:
                arr[idx, BASES[ch], i] = 1
    return arr


def rf_features(seqs):
    onehot = one_hot_4x20(seqs).reshape(len(seqs), -1)
    gc = np.array([[gc_content(s)] for s in seqs])
    return np.hstack([onehot, gc])


def gene_level_split(df, test_gene_frac=0.2, seed=42):
    """Split by gene: 80% of genes -> train, 20% -> test.
    Returns (train_guides, val_guides, test_guides, split_name)."""
    genes = df['Gene_Symbol'].unique().tolist()
    tr_genes, te_genes = train_test_split(genes, test_size=test_gene_frac, random_state=seed)
    train = df[df['Gene_Symbol'].isin(tr_genes)]
    test = df[df['Gene_Symbol'].isin(te_genes)]
    # val from train GUIDES (80/20 of train guides)
    train, val = train_test_split(train, test_size=0.2, random_state=seed)
    split_name = (f"gene-level ({len(tr_genes)} train genes / {len(te_genes)} test genes)")
    return train, val, test, split_name


def train_rf(train, test):
    Xtr = rf_features(train['Sequence'].tolist())
    ytr = train['Activity'].values
    Xte = rf_features(test['Sequence'].tolist())
    yte = test['Activity'].values
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    rho, _ = spearmanr(yte, pred)
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    return {'model': 'RandomForest', 'spearman': round(rho, 3), 'rmse': round(rmse, 3)}


def train_cnn(train, val, test, epochs=15):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except Exception as e:
        print(f"[CNN] skipped (torch not installed): {e}")
        return None

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

    Xtr = torch.tensor(one_hot_4x20(train['Sequence'].tolist()), dtype=torch.float32)
    ytr = torch.tensor(train['Activity'].values, dtype=torch.float32).view(-1, 1)
    Xva = torch.tensor(one_hot_4x20(val['Sequence'].tolist()), dtype=torch.float32)
    yva = torch.tensor(val['Activity'].values, dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(one_hot_4x20(test['Sequence'].tolist()), dtype=torch.float32)
    yte = torch.tensor(test['Activity'].values, dtype=torch.float32).view(-1, 1)

    model = SimpleCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

    best_rho = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy().flatten()
        rho_v, _ = spearmanr(yva.numpy().flatten(), pv)
        if rho_v > best_rho:
            best_rho = rho_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  CNN epoch {epoch+1}/{epochs} train_loss={loss.item():.4f} val_spearman={rho_v:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy().flatten()
    rho, _ = spearmanr(yte.numpy().flatten(), pred)
    rmse = float(np.sqrt(np.mean((yte.numpy().flatten() - pred) ** 2)))
    print(f"[CNN 1D]  test Spearman={rho:.3f} RMSE={rmse:.3f} (restored best val epoch)")
    return {'model': 'CNN', 'spearman': round(rho, 3), 'rmse': round(rmse, 3)}


def run(df, tag, out_path, do_cnn=True):
    train, val, test, split_name = gene_level_split(df)
    print(f"\n[{tag}] {split_name}")
    print(f"Train genes: {df['Gene_Symbol'].isin(train['Gene_Symbol']).sum() if False else len(train['Gene_Symbol'].unique())} | "
          f"Train guides: {len(train)} | Val: {len(val)} | Test genes: {len(test['Gene_Symbol'].unique())} | Test guides: {len(test)}")
    # verify no leakage
    tr_genes = set(train['Gene_Symbol']); te_genes = set(test['Gene_Symbol'])
    assert tr_genes.isdisjoint(te_genes), "LEAK: train/test gene overlap!"
    print("No gene leakage between train/test ✅")

    results = [train_rf(train, test)]
    if do_cnn:
        cnn = train_cnn(train, val, test)
        if cnn: results.append(cnn)

    rows = pd.DataFrame([{
        'split': split_name, 'tag': tag,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'n_train_genes': len(tr_genes), 'n_test_genes': len(te_genes),
        **r} for r in results])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if os.path.exists(out_path):
        pd.concat([pd.read_csv(out_path), rows]).to_csv(out_path, index=False)
    else:
        rows.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='results/v2/gene_labels.csv')
    ap.add_argument('--out', default='results/v2')
    ap.add_argument('--no-cnn', action='store_true', help='skip CNN (RF only)')
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} mapped guides, {df['Gene_Symbol'].nunique()} genes")

    # Primary: all 54,956
    run(df, 'V2A_all', os.path.join(args.out, 'v2a_metrics.csv'), do_cnn=not args.no_cnn)

    # Secondary: genes with >=5 guides (guard: skip if none qualify)
    counts = df['Gene_Symbol'].value_counts()
    keep = counts[counts >= 5].index
    df5 = df[df['Gene_Symbol'].isin(keep)]
    if len(df5) > 0 and len(keep) > 1:
        print(f"\nSecondary: {len(df5)} guides across {len(keep)} genes (>=5 guides)")
        run(df5, 'V2A_genes5plus', os.path.join(args.out, 'v2a_metrics.csv'), do_cnn=not args.no_cnn)
    else:
        print("\nSecondary: no genes with >=5 guides - skipped")

    print("\nDone. Compare V2A vs V1: random-split RF 0.728 / CNN 0.745 (from results/eval_metrics.csv)")
