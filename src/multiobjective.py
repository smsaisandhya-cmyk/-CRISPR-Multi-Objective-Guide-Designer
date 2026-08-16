import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def is_pareto_efficient(costs, maximize=True):
    if maximize: costs = -costs
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < costs[i], axis=1)
            is_efficient[i] = True
    return is_efficient

path='results/multi_objective_scores.csv'
if not os.path.exists(path):
    import subprocess, sys
    subprocess.run([sys.executable, 'src/features.py'])
df=pd.read_csv(path)
print(f"Loaded {len(df)} guides")
eff = df['eff_proxy'].values
off_inv = 1 - df['off_target_risk'].values
struct = df['struct_score'].values
costs = np.vstack([eff, off_inv, struct]).T
pareto_mask = is_pareto_efficient(costs, maximize=True)
df['pareto'] = pareto_mask
print(f"Pareto-optimal guides: {pareto_mask.sum()} / {len(df)}")
print(df[pareto_mask][['guide','eff_proxy','off_target_risk','struct_score']].to_string())
os.makedirs('results/figures', exist_ok=True)
colors = ['red' if p else 'steelblue' for p in pareto_mask]
sizes = [120 if p else 45 for p in pareto_mask]
plt.figure(figsize=(9,6))
scatter = plt.scatter(off_inv, eff, c=struct, cmap='RdYlGn', s=sizes, edgecolors=colors, linewidth=1.2, alpha=0.85)
plt.colorbar(scatter, label='Structure score (higher=better)')
plt.xlabel('Specificity (1 - off_target_risk) -> higher = fewer off-targets')
plt.ylabel('Predicted Efficiency')
plt.title('CRISPR Guides: Pareto Front (efficiency vs specificity)\nRed border = Pareto-optimal')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('results/figures/pareto_2d.png', dpi=180)
print("Saved results/figures/pareto_2d.png")
df.to_csv('results/pareto_ranked.csv', index=False)
print("Saved results/pareto_ranked.csv")