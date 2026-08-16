import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
BASES = {'A':0,'C':1,'G':2,'T':3,'U':3}
def one_hot(seq):
    arr = np.zeros((4, len(seq)))
    for i,ch in enumerate(seq.upper()):
        if ch in BASES:
            arr[BASES[ch], i] = 1
    return arr.flatten()
def gc_content(seq):
    return (seq.upper().count('G')+seq.upper().count('C'))/len(seq) if len(seq)>0 else 0
def featurize_df(df, seq_col):
    X_seq = np.vstack([one_hot(s) for s in df[seq_col]])
    gc = np.array([[gc_content(s)] for s in df[seq_col]])
    return np.hstack([X_seq, gc])
def load_or_synth(data_path, seq_col, eff_col):
    import os
    if data_path and os.path.exists(data_path):
        print(f"Loading real data from {data_path}")
        df = pd.read_csv(data_path) if not data_path.endswith('.xlsx') else pd.read_excel(data_path)
        print(f"Columns found: {df.columns.tolist()[:10]}")
        df = df[[seq_col, eff_col]].dropna()
        df.columns = ['seq','eff']
        df = df[df['seq'].str.len()>=19]
        print(f"Loaded {len(df)} guides")
        return df
    else:
        print("No real data -> synthetic")
        np.random.seed(42)
        seqs = [''.join(np.random.choice(list('ACGT'),20)) for _ in range(1800)]
        effs=[np.clip(0.4+0.4*(1-2*abs(gc_content(s)-0.5))+0.1*(s[19]=='G')+np.random.normal(0,0.15),0,1) for s in seqs]
        return pd.DataFrame({'seq':seqs,'eff':effs})
def train_and_eval(df):
    X = featurize_df(df,'seq')
    y = df['eff'].values
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train,y_train)
    pred = rf.predict(X_test)
    rho,_ = spearmanr(y_test,pred)
    rmse = float(np.sqrt(np.mean((y_test-pred)**2)))
    print(f"\n[RandomForest] Spearman={rho:.3f} RMSE={rmse:.3f}")
    try:
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        def seq_to_4x20(seqs):
            arr=np.zeros((len(seqs),4,20))
            for idx,s in enumerate(seqs):
                for i,ch in enumerate(s[:20]):
                    if ch in BASES: arr[idx,BASES[ch],i]=1
            return arr
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
        Xtr = torch.tensor(seq_to_4x20(df.iloc[train_idx]['seq']), dtype=torch.float32)
        ytr = torch.tensor(df.iloc[train_idx]['eff'].values, dtype=torch.float32).view(-1,1)
        Xte = torch.tensor(seq_to_4x20(df.iloc[test_idx]['seq']), dtype=torch.float32)
        yte = torch.tensor(df.iloc[test_idx]['eff'].values, dtype=torch.float32).view(-1,1)
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(nn.Conv1d(4,32,3,padding=1), nn.ReLU(), nn.Conv1d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
                self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
            def forward(self,x): return self.fc(self.conv(x))
        model=SimpleCNN()
        opt=torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn=nn.MSELoss()
        loader=DataLoader(TensorDataset(Xtr,ytr), batch_size=64, shuffle=True)
        for epoch in range(12):
            model.train()
            for xb,yb in loader:
                opt.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward(); opt.step()
            if (epoch+1)%4==0: print(f"  CNN epoch {epoch+1}/12 loss={loss.item():.4f}")
        model.eval()
        with torch.no_grad(): pred3=model(Xte).numpy().flatten()
        rho3,_=spearmanr(yte.numpy().flatten(), pred3)
        print(f"[CNN 1D]      Spearman={rho3:.3f} (trained 12 epochs on CPU)")
        import os
        os.makedirs('results', exist_ok=True)
        torch.save(model.state_dict(), 'results/cnn_baseline.pt')
        print("Saved CNN to results/cnn_baseline.pt")
    except Exception as e:
        print(f"[CNN] skipped: {e}")
        import traceback; traceback.print_exc()
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/doench2016.xlsx')
    ap.add_argument('--seq-col', default='sgRNA Sequence')
    ap.add_argument('--eff-col', default='Activity')
    args=ap.parse_args()
    df=load_or_synth(args.data, args.seq_col, args.eff_col)
    print(df.head())
    import os
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/guides_used.csv', index=False)
    train_and_eval(df)
    print("\nDone.")
