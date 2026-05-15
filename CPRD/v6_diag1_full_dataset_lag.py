"""
Task 1: 全数据集 (train + val + test) dementia 患者 lag 分布统计.
Lag = 最后一个事件 (dementia 诊断) - 倒数第二个事件 (最后非 dementia 记录) 的天数, 换算成年.
"""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEM_CODES = set([
    "F110.","Eu00.","Eu01.","Eu02z","Eu002","E00..","Eu023","Eu00z","Eu025","Eu01z",
    "E001.","F1100","Eu001","E004.","Eu000","Eu02.","Eu013","E000.","Eu01y","E001z",
    "F1101","Eu020","E004z","E0021","Eu02y","Eu012","Eu011","E00z.","E0040","E003.","E0020"
])
OUT_PNG = '/Data0/swangek_data/991/CPRD/figs/full_dataset_dementia_lag.png'

all_lags = []
splits_data = {}
for split in ['train', 'val', 'test']:
    pattern = f'/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_aug_v2/split={split}/**/*.parquet'
    files = sorted(glob.glob(pattern, recursive=True))
    split_lags = []
    for fp in tqdm(files, desc=split, leave=False):
        df = pq.read_table(fp).to_pandas()
        for _, row in df.iterrows():
            ev = list(row['EVENT']); dt = list(row['DATE'])
            if len(ev) < 2: continue
            if str(ev[-1]) not in DEM_CODES: continue
            if str(ev[-2]) in DEM_CODES: continue
            last = pd.Timestamp(dt[-1]).to_pydatetime()
            second = pd.Timestamp(dt[-2]).to_pydatetime()
            lag_y = (last - second).days / 365.25
            if lag_y > 0:
                split_lags.append(lag_y)
    splits_data[split] = np.array(split_lags)
    all_lags.extend(split_lags)

all_lags = np.array(all_lags)

print("\n" + "="*70)
print("全数据集 Dementia 确诊 lag 分布 (n={})".format(len(all_lags)))
print("="*70)
for split, arr in splits_data.items():
    print(f"\n{split}: n={len(arr)}")
    print(f"  Mean   = {np.mean(arr):.3f} 年")
    print(f"  Median = {np.median(arr):.3f} 年")
    print(f"  P25    = {np.percentile(arr, 25):.3f} 年")
    print(f"  P75    = {np.percentile(arr, 75):.3f} 年")

print(f"\n--- 全数据集合并 (n={len(all_lags)}) ---")
print(f"  Mean   = {np.mean(all_lags):.3f} 年")
print(f"  Median = {np.median(all_lags):.3f} 年")
print(f"  Std    = {np.std(all_lags):.3f} 年")
print(f"  Min    = {np.min(all_lags):.3f} 年")
print(f"  Max    = {np.max(all_lags):.3f} 年")
print(f"\n  Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p:2d}    = {np.percentile(all_lags, p):.3f} 年")

print(f"\n  累计比例 (在某年内被诊断):")
for cap in [0.5, 1, 2, 3, 5, 7, 10, 15, 20]:
    pct = (all_lags <= cap).mean() * 100
    print(f"    ≤ {cap:4.1f} 年: {pct:5.1f}% ({(all_lags <= cap).sum():4d} 患者)")

# 画图
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# (a) Train/val/test 叠加直方图
ax = axes[0, 0]
colors = {'train': 'steelblue', 'val': 'orange', 'test': 'green'}
for split, arr in splits_data.items():
    ax.hist(arr, bins=60, alpha=0.6, label=f'{split} (n={len(arr)})', color=colors[split])
ax.set_xlabel('Lag (年): dementia 诊断 - 上次非 dementia 记录')
ax.set_ylabel('患者数')
ax.set_title('各 split 分布对比')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 18)

# (b) 全数据集 + percentile lines
ax = axes[0, 1]
ax.hist(all_lags, bins=80, edgecolor='black', color='steelblue')
ax.axvline(np.median(all_lags), color='red', linestyle='--', linewidth=2, label=f'Median = {np.median(all_lags):.2f}y')
ax.axvline(np.mean(all_lags), color='orange', linestyle='--', linewidth=2, label=f'Mean = {np.mean(all_lags):.2f}y')
ax.axvline(5, color='gray', linestyle=':', label='5y mark')
ax.axvline(10, color='gray', linestyle=':', label='10y mark')
ax.set_xlabel('Lag (年)')
ax.set_ylabel('患者数')
ax.set_title(f'全数据集 (n={len(all_lags)})')
ax.legend()
ax.grid(alpha=0.3)

# (c) 累计分布
ax = axes[1, 0]
sorted_lags = np.sort(all_lags)
cum_fraction = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)
ax.plot(sorted_lags, cum_fraction, linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axvline(np.median(all_lags), color='red', linestyle='--', alpha=0.5)
ax.axvline(5, color='gray', linestyle=':', label='5y')
ax.axvline(10, color='gray', linestyle=':', label='10y')
ax.set_xlabel('Lag (年)')
ax.set_ylabel('累计比例')
ax.set_title('累计分布 (CDF)')
ax.legend()
ax.grid(alpha=0.3)

# (d) Box plot per split
ax = axes[1, 1]
data_box = [splits_data['train'], splits_data['val'], splits_data['test']]
ax.boxplot(data_box, labels=['train', 'val', 'test'])
ax.set_ylabel('Lag (年)')
ax.set_title('每个 split 的 box plot')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120)
print(f"\nFigure saved to {OUT_PNG}")
