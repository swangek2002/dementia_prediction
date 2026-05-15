"""
Task 2: 展示模型对真实患者的完整预测输出.
从 V5 NPZ 选 4 个 representative 患者 (dementia 早期, dementia 晚期, death, censored),
画他们的 CIF curves over [0, 25 年], 标出各个时间点的预测概率.
"""
import numpy as np
import pyarrow.parquet as pq
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load V5 inference NPZ
v5 = np.load('/Data0/swangek_data/991/CPRD/data/test_cif_v5_full.npz', allow_pickle=True)
pids = v5['patient_ids']
labels = v5['labels']
event_time_scaled = v5['event_time_scaled']
cif_dementia = v5['cif_dementia']
cif_death = v5['cif_death']
t_eval = v5['t_eval']

# Conversion: event_time_scaled * 25 = actual years
t_eval_years = t_eval * 25
event_time_years = event_time_scaled * 25

print("="*70)
print("Model output format")
print("="*70)
print(f"Number of test patients: {len(pids)}")
print(f"t_eval shape: {t_eval.shape}  (normalized [0,1])")
print(f"t_eval in years: {t_eval_years[0]:.2f} to {t_eval_years[-1]:.2f}")
print(f"cif_dementia shape per patient: ({t_eval.shape[0]},)  (1000 time points)")
print(f"cif_death shape per patient: ({t_eval.shape[0]},)  (1000 time points)")
print()

# Pick 4 representative patients
np.random.seed(42)

# Find a dementia patient with early event (event_time < 0.1 = 2.5y)
early_dem_idx = np.where((labels == 'dementia') & (event_time_scaled < 0.1))[0]
# Find a dementia patient with late event (event_time > 0.4 = 10y)
late_dem_idx = np.where((labels == 'dementia') & (event_time_scaled > 0.4))[0]
# Find a death patient (mid time)
death_idx = np.where((labels == 'death') & (event_time_scaled > 0.15) & (event_time_scaled < 0.35))[0]
# Find a censored patient (mid time)
cens_idx = np.where((labels == 'censored') & (event_time_scaled > 0.15) & (event_time_scaled < 0.35))[0]

example_indices = {
    'Dementia (early)': np.random.choice(early_dem_idx),
    'Dementia (late)': np.random.choice(late_dem_idx),
    'Death': np.random.choice(death_idx),
    'Censored': np.random.choice(cens_idx),
}

print("Selected example patients:")
for label, idx in example_indices.items():
    pid = pids[idx]
    actual_label = labels[idx]
    event_t_norm = event_time_scaled[idx]
    event_t_yr = event_t_norm * 25
    cif_dem_final = cif_dementia[idx, -1]
    cif_dth_final = cif_death[idx, -1]
    print(f"\n{label}:  PID={pid}")
    print(f"  Ground truth label: {actual_label}")
    print(f"  Actual event time (normalized): {event_t_norm:.4f}")
    print(f"  Actual event time (years): {event_t_yr:.2f}")
    print(f"  Final CIF_dementia (t=25y): {cif_dem_final:.4f}")
    print(f"  Final CIF_death (t=25y):    {cif_dth_final:.4f}")

# 详细打印每个 example 的预测在关键时间点
print("\n" + "="*70)
print("Model's CIF predictions at specific time horizons (each patient)")
print("="*70)
key_horizons_y = [1, 2, 3, 5, 7, 10, 14, 20, 25]
key_horizons_norm = [y/25 for y in key_horizons_y]
key_indices = [np.argmin(np.abs(t_eval - h)) for h in key_horizons_norm]

for label, idx in example_indices.items():
    pid = pids[idx]
    print(f"\n{label}  PID={pid}  Actual: {labels[idx]} at {event_time_scaled[idx]*25:.2f}y")
    print(f"  {'Horizon':>10s} {'CIF Dementia':>14s} {'CIF Death':>14s} {'CIF Combined':>14s}")
    for h_y, h_idx in zip(key_horizons_y, key_indices):
        cd = cif_dementia[idx, h_idx]
        ct = cif_death[idx, h_idx]
        print(f"  {h_y:>8d}y  {cd:>14.4f} {ct:>14.4f} {cd+ct:>14.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, (label, idx) in zip(axes.flatten(), example_indices.items()):
    pid = pids[idx]
    actual_label = labels[idx]
    event_t_yr = event_time_scaled[idx] * 25

    ax.plot(t_eval_years, cif_dementia[idx], color='red', linewidth=2, label='CIF Dementia')
    ax.plot(t_eval_years, cif_death[idx], color='blue', linewidth=2, label='CIF Death')
    ax.plot(t_eval_years, cif_dementia[idx] + cif_death[idx], color='gray',
            linewidth=1, linestyle='--', alpha=0.7, label='CIF Combined (dementia+death)')
    ax.plot(t_eval_years, 1 - (cif_dementia[idx] + cif_death[idx]), color='green',
            linewidth=1, linestyle=':', alpha=0.7, label='Survival probability')

    # Mark the actual event time
    if actual_label != 'censored':
        ax.axvline(event_t_yr, color='orange', linestyle='-.', linewidth=2,
                   label=f'Actual {actual_label} at {event_t_yr:.2f}y')
    else:
        ax.axvline(event_t_yr, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Last observed at {event_t_yr:.2f}y (censored)')

    # Reference vertical lines
    for h in [5, 10]:
        ax.axvline(h, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('Time from prediction point (years)')
    ax.set_ylabel('Cumulative Incidence Function (CIF)')
    ax.set_title(f'{label} — PID {pid}\nActual outcome: {actual_label}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
out = '/Data0/swangek_data/991/CPRD/figs/real_prediction_examples.png'
plt.savefig(out, dpi=120)
print(f"\nFigure saved to {out}")
