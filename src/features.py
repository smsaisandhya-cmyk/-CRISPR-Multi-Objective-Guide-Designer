import numpy as np, pandas as pd, os
try:
    import RNA
    HAS_VIENNA=True
except:
    HAS_VIENNA=False

def gc_score(seq):
    gc = (seq.count('G')+seq.count('C'))/len(seq)
    return 1.0 if 0.4 <= gc <= 0.65 else max(0, 1 - 2*abs(gc-0.52))

def poly_penalty(seq):
    penalty=0
    if 'TTTT' in seq: penalty+=1
    if 'GGGG' in seq: penalty+=0.5
    return penalty

def structure_features(seq):
    if HAS_VIENNA:
        (ss, mfe) = RNA.fold(seq)
        return {'mfe': mfe, 'struct_score': 1 if mfe > -4 else max(0, 1 - (abs(mfe)-4)/8)}
    else:
        mfe = - (seq.count('G')*0.2 + seq.count('C')*0.2)*2.5
        return {'mfe': mfe, 'struct_score': 1 if mfe > -4 else 0.5}

def off_target_proxy(seq):
    import numpy as np
    np.random.seed(hash(seq) % 2**32)
    base = 0.15 + 0.1*(seq.count('G')/len(seq)) + np.random.rand()*0.25
    if len(set(seq[-8:]))<=2: base+=0.2
    return float(np.clip(base,0,1))

def score_guides(df, seq_col='seq'):
    rows=[]
    for s in df[seq_col]:
        s=s.upper().replace('U','T')
        gc = (s.count('G')+s.count('C'))/len(s)
        gcs = gc_score(s)
        struct = structure_features(s)
        off = off_target_proxy(s)
        eff_proxy = 0.5 + 0.3*gcs + 0.2*struct['struct_score'] - 0.15*off + np.random.normal(0,0.07)
        rows.append({'guide': s, 'GC': round(gc,3), 'GC_score': round(gcs,3), 'MFE': round(struct['mfe'],2), 'struct_score': round(struct['struct_score'],3), 'off_target_risk': round(off,3), 'eff_proxy': round(float(np.clip(eff_proxy,0,1)),3)})
    return pd.DataFrame(rows)

if __name__=='__main__':
    if os.path.exists('results/guides_used.csv'):
        df=pd.read_csv('results/guides_used.csv').head(15)
        df=df.rename(columns={'seq':'seq'})
    else:
        import random
        seqs=[''.join(random.choice('ACGT') for _ in range(20)) for _ in range(15)]
        df=pd.DataFrame({'seq':seqs})
    scored=score_guides(df)
    print(scored.to_string())
    os.makedirs('results', exist_ok=True)
    scored.to_csv('results/multi_objective_scores.csv', index=False)
    print("\nSaved to results/multi_objective_scores.csv")