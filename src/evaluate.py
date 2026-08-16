"""
evaluate.py - honest, leakage-safe evaluation for CRISPR efficiency models.

Trains RandomForest + CNN on TRAIN only, reports Spearman on a TEST set
that the model never saw during training or tuning.

Random split (default): 80% train / 10% val / 10% test, fixed seed 42.
Gene split (--gene-col): all guides from 80% of genes go to train,
all guides from the other 20% of genes go to test - no same-gene leakage.

Usage:
  py src/evaluate.py --data data/deepHF_clean.csv --seq-col Sequence --eff-col Activity
  py src/evaluate.py --data data/doench_real_gene.csv --seq-col Sequence --eff-col Activity --gene-col Gene
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
    """(n, 4, 20) array - input for CNN."""
    arr = np.zeros((len(seqs), 4, 20))
    for idx, s in enumerate(seqs):
        for i, ch in enumerate(s[:20]):
            if ch in BASES:
                arr[idx, BASES[ch], i] = 1
    return arr


def rf_features(seqs):
    """flat one-hot (80) + GC (1) - input for RandomForest."""
    onehot = one_hot_4x20(seqs).reshape(len(seqs), -1)
    gc = np.array([[gc_content(s)] for s in seqs])
    return np.hstack([onehot, gc])


def load_data(args):
    if args.data.endswith('.xlsx'):
        df = pd.read_excel(args.data, sheet_name=0)
    else:
        df = pd.read_csv(args.data)
    cols = [args.seq_col, args.eff_col]
    if args.gene_col:
        cols.append(args.gene_col)
    df = df[cols].dropna()
    df.columns = ['seq', 'eff'] + (['gene'] if args.gene_col else [])
    df = df[df['seq'].str.len() >= 19]
    before = len(df)
    df = df.drop_duplicates(subset='seq')
    print(f"Guides after filter: {before} -> after removing exact duplicates: {len(df)}")
    return df


def make_splits(df, args):
    if args.gene_col:
        genes = df['gene'].unique().tolist()
        tr_genes, te_genes = train_test_split(genes, test_size=0.2, random_state=42)
        train = df[df['gene'].isin(tr_genes)]
        test = df[df['gene'].isin(te_genes)]
        train, val = train_test_split(train, test_size=0.125, random_state=42)
        split_name = f"gene-level ({len(tr_genes)} train genes / {len(te_genes)} test genes)"
    else:
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        split_name = "random 80/10/10 (seeded)"
    return train, val, test, split_name


def train_rf(train, val, test):
    Xtr = rf_features(train['seq'].tolist())
    ytr = train['eff'].values
    Xte = rf_features(test['seq'].tolist())
    yte = test['eff'].values
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    rho, _ = spearmanr(yte, pred)
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    print(f"[RandomForest] test Spearman={rho:.3f} RMSE={rmse:.3f}")
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

    Xtr = torch.tensor(one_hot_4x20(train['seq'].tolist()), dtype=torch.float32)
    ytr = torch.tensor(train['eff'].values, dtype=torch.float32).view(-1, 1)
    Xva = torch.tensor(one_hot_4x20(val['seq'].tolist()), dtype=torch.float32)
    yva = torch.tensor(val['eff'].values, dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(one_hot_4x20(test['seq'].tolist()), dtype=torch.float32)
    yte = torch.tensor(test['eff'].values, dtype=torch.float32).view(-1, 1)

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
            print(f"  CNN epoch {epoch + 1}/{epochs} train_loss={loss.item():.4f} val_spearman={rho_v:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy().flatten()
    rho, _ = spearmanr(yte.numpy().flatten(), pred)
    rmse = float(np.sqrt(np.mean((yte.numpy().flatten() - pred) ** 2)))
    print(f"[CNN 1D]  test Spearman={rho:.3f} RMSE={rmse:.3f} (restored best val epoch)")
    return {'model': 'CNN', 'spearman': round(rho, 3), 'rmse': round(rmse, 3)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--seq-col', default='Sequence')
    ap.add_argument('--eff-col', default='Activity')
    ap.add_argument('--gene-col', default=None)
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()

    df = load_data(args)
    train, val, test, split_name = make_splits(df, args)
    print(f"\nDataset: {args.data}")
    print(f"Split: {split_name}")
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}  (test set untouched)")

    results = [train_rf(train, val, test)]
    cnn = train_cnn(train, val, test, epochs=args.epochs)
    if cnn is not None:
        results.append(cnn)

    os.makedirs('results', exist_ok=True)
    rows = pd.DataFrame([{
        'dataset': os.path.basename(args.data),
        'split': split_name,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        **r
    } for r in results])
    out = 'results/eval_metrics.csv'
    if os.path.exists(out):
        pd.concat([pd.read_csv(out), rows]).to_csv(out, index=False)
    else:
        rows.to_csv(out, index=False)
    print(f"\nSaved metrics to {out}")
    print(rows.to_string(index=False))