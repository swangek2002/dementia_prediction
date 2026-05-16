"""
Generate figures for the English progress report.

Models included (clean / acceptable baselines):
  - Single-GP + HESstatic
  - Dual-gated + HESstatic
  - Dual + HESstatic + SST1%
  - Dual + HESstatic + SST2%
  - Dual + HESstatic + SST5%
  - Single-GP + HESstatic + leak-fix (V6 base, just trained)
  - GP-only 5-fold CV @ idx 68 (different cohort — included as no-HES baseline)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path('/Data0/swangek_data/991/CPRD/figs')
FIG_DIR.mkdir(exist_ok=True)

# =================================================================
# Common model data
# =================================================================
# All clean models + idx68 5-fold CV (added as baseline)
models_short = [
    'GP-only 5-fold CV\n(idx68 baseline)',
    'Single-GP\n+ HESstatic',
    'Dual-gated\n+ HESstatic',
    'Dual + HESstatic\n+ SST1%',
    'Dual + HESstatic\n+ SST2%',
    'Dual + HESstatic\n+ SST5%',
    'Single-GP + HESstatic\n+ leak-fix',
]
# Cohort C_td (dementia): for idx68 CV = mean of 5 folds on own per-fold test sets
ctd_dem  = [0.8240, 0.8451, 0.8447, 0.8506, 0.8487, 0.8467, 0.8443]
ctd_dem_err = [0.027,  0,      0,      0,      0,      0,      0]
# 5y AUROC: idx68 CV mean = 0.8108 ± 0.031 (own per-fold cohort, prev ~0.32%)
auroc_5y = [0.8108, 0.9236, 0.9284, 0.9271, 0.9288, 0.9219, 0.9246]
auroc_err = [0.031,  0,      0,      0,      0,      0,      0]
# PPV @ top 1%: idx68 CV mean = 1.43% (5y, exclude_early_censored, own per-fold cohort, prev 0.32%)
ppv_1pct = [1.43,   68.89,  71.11,  55.56,  62.22,  53.33,  64.44]
ppv_1pct_err = [0.96, 0, 0, 0, 0, 0, 0]
ppv_5pct = [1.22,   27.88,  27.88,  29.65,  28.76,  30.97,  30.09]
ppv_10pct= [0.93,   19.03,  19.25,  19.25,  18.81,  18.58,  18.81]

# Color palette: idx68 CV = grey (different cohort caveat), single-baseline = blue,
# dual = blue, SST = green, leak-fix = orange
colors = ['#9c9c9c', '#5b8fb9', '#5b8fb9', '#2e7d32', '#2e7d32', '#2e7d32', '#e67e22']

# =================================================================
# Fig 1: All-clean-model comparison (now with idx68 CV + leak-fix)
# =================================================================
x = np.arange(len(models_short))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 13), gridspec_kw={'hspace': 0.35})

# Panel 1: C_td dementia
ax1.bar(x, ctd_dem, yerr=ctd_dem_err, color=colors, edgecolor='black', linewidth=0.5,
        capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
for i, (v, e) in enumerate(zip(ctd_dem, ctd_dem_err)):
    suffix = f'\n±{e:.3f}' if e > 0 else ''
    ax1.text(i, v + 0.004, f'{v:.4f}{suffix}', ha='center', fontsize=9)
ax1.set_ylim(0.78, 0.87)
ax1.set_ylabel('Cohort C_td (Antolini) — dementia', fontsize=11)
ax1.set_title('Cohort-Level C_td (Antolini), Dementia', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(x); ax1.set_xticklabels([])

# Panel 2: AUROC
ax2.bar(x, auroc_5y, yerr=auroc_err, color=colors, edgecolor='black', linewidth=0.5,
        capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
for i, (v, e) in enumerate(zip(auroc_5y, auroc_err)):
    suffix = f'\n±{e:.3f}' if e > 0 else ''
    ax2.text(i, v + 0.005, f'{v:.4f}{suffix}', ha='center', fontsize=9)
ax2.set_ylim(0.78, 0.95)
ax2.set_ylabel('5y AUROC', fontsize=11)
ax2.set_title('5y AUROC — exclude_early_censored policy', fontsize=11)
ax2.axhline(0.5, color='red', linestyle='--', linewidth=0.6, alpha=0.4)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(x); ax2.set_xticklabels([])

# Panel 3: PPV @ top 1%
ax3.bar(x, ppv_1pct, yerr=ppv_1pct_err, color=colors, edgecolor='black', linewidth=0.5,
        capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
for i, (v, e) in enumerate(zip(ppv_1pct, ppv_1pct_err)):
    suffix = f'\n±{e:.2f}' if e > 0 else ''
    ax3.text(i, v + 1.5, f'{v:.1f}%{suffix}', ha='center', fontsize=9)
ax3.set_ylim(0, 80)
ax3.set_ylabel('5y PPV @ top 1% (%)', fontsize=11)
ax3.set_title('5y PPV at top 1% — clinical screening yield\n'
              '(idx68 CV has much lower 5y prevalence ~0.32% vs ~2.68% for idx72 models — bars not directly comparable)',
              fontsize=10)
ax3.axhline(2.68, color='red', linestyle='--', linewidth=0.6, label='idx72 cohort prevalence 2.68%')
ax3.axhline(0.32, color='gray', linestyle=':', linewidth=0.6, label='idx68 cohort prevalence 0.32%')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(models_short, rotation=0, ha='center', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = FIG_DIR / 'fig_model_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# =================================================================
# Fig 2: TRADE comparison — ALL our models vs TRADE (AUROC from 0.5)
# =================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                  gridspec_kw={'width_ratios': [1, 1.4]})

# Build the bars: 7 our models + 1 TRADE = 8
all_labels = models_short + ['TRADE\n(NYU)']
all_auroc = auroc_5y + [0.735]
all_auroc_err = auroc_err + [0]
all_ppv1 = ppv_1pct + [39.2]
all_ppv5 = ppv_5pct + [27.8]
all_ppv10 = ppv_10pct + [22.4]
all_colors = colors + ['#d36b6b']

# Panel A: AUROC — y-axis from 0.5 to 1.0 (random = 0.5 baseline)
xA = np.arange(len(all_labels))
ax_a.bar(xA, all_auroc, yerr=all_auroc_err, color=all_colors, edgecolor='black',
         linewidth=0.5, capsize=4)
for i, (v, e) in enumerate(zip(all_auroc, all_auroc_err)):
    suffix = f'\n±{e:.3f}' if e > 0 else ''
    ax_a.text(i, v + 0.008, f'{v:.3f}{suffix}', ha='center', fontsize=9)
ax_a.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Random baseline (0.5)')
ax_a.set_ylim(0.5, 1.0)
ax_a.set_xticks(xA)
ax_a.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8.5)
ax_a.set_ylabel('5y AUROC')
ax_a.set_title('5y AUROC — All our models vs TRADE\n(y-axis starts at 0.5 = random baseline)')
ax_a.legend(loc='lower right', fontsize=8.5)
ax_a.grid(axis='y', alpha=0.3)

# Panel B: PPV — grouped bars (top 1%, 5%, 10%)
metrics = ['PPV @ top 1%', 'PPV @ top 5%', 'PPV @ top 10%']
xB = np.arange(len(metrics))
w = 0.10
for i, (lbl, color) in enumerate(zip(all_labels, all_colors)):
    vals = [all_ppv1[i], all_ppv5[i], all_ppv10[i]]
    pos = xB + (i - len(all_labels)/2 + 0.5) * w
    ax_b.bar(pos, vals, w, label=lbl.replace('\n', ' '), color=color,
             edgecolor='black', linewidth=0.4)

ax_b.set_xticks(xB)
ax_b.set_xticklabels(metrics)
ax_b.set_ylabel('PPV (%)')
ax_b.set_ylim(0, max(all_ppv1+all_ppv5+all_ppv10)+15)
ax_b.set_title('PPV @ top X% — All our models vs TRADE\n'
               '(Note: idx68 CV has 0.32% baseline prevalence vs 2.68% for idx72 models)',
               fontsize=10)
ax_b.legend(loc='upper right', fontsize=7.5, ncol=2)
ax_b.grid(axis='y', alpha=0.3)
ax_b.axhline(2.68, color='red', linestyle=':', linewidth=0.6, alpha=0.6, label='_idx72 prev 2.68%')
ax_b.axhline(7.2, color='darkred', linestyle=':', linewidth=0.6, alpha=0.6, label='_TRADE prev 7.2%')
ax_b.axhline(0.32, color='gray', linestyle=':', linewidth=0.6, alpha=0.6, label='_idx68 prev 0.32%')

plt.suptitle('Direct Comparison: All our (clean) models + idx68 5-fold CV baseline vs TRADE', fontsize=12)
plt.tight_layout()
out = FIG_DIR / 'fig_trade_compare.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# =================================================================
# Fig 3: Code-set overlap — stacked-bar (unchanged from before)
# =================================================================
fig, ax = plt.subplots(figsize=(10, 4))
ours = 5996; both = 5996; trade_only = 174
ax.barh(1, ours, left=0, color='#5b8fb9', edgecolor='black', linewidth=0.5, label='Ours ∩ TRADE-strict')
ax.barh(0, both, left=0, color='#5b8fb9', edgecolor='black', linewidth=0.5)
ax.barh(0, trade_only, left=both, color='#d36b6b', edgecolor='black', linewidth=0.5,
        label='TRADE-strict only (G31.0 + G31.8 + G31.1)')
ax.text(ours/2, 1, '5,996  (4.50% of cohort)', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold')
ax.text(both/2, 0, '5,996', ha='center', va='center', fontsize=10,
        color='white', fontweight='bold')
ax.text(both + trade_only/2, 0, '174', ha='center', va='center', fontsize=9,
        color='white', fontweight='bold')
ax.set_yticks([1, 0])
ax.set_yticklabels(['Ours\n(F00/F01/F02/F03/G30)', 'TRADE-strict\n(+ G31.0 + G31.8 + G31.1)'])
ax.set_xlabel('Number of unique patients in UKB cohort (N=133,322)')
ax.set_xlim(0, ours + trade_only + 600)
ax.set_title('HES ICD-10 code-set comparison: ours is a strict subset of TRADE-strict\n'
             'Ours-only = 0 patients; TRADE catches 174 additional patients (FTD/Lewy/MCI-G31.1)',
             fontsize=11)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
out = FIG_DIR / 'fig_code_overlap.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# =================================================================
# Fig 4: SST progression (unchanged from previous)
# =================================================================
rounds = ['no SST\n(Dual + HESstatic)', '+ SST 1%\n(+ 771 pseudo)', '+ SST 2%\n(+ 1,595)', '+ SST 5%\n(+ 3,814)']
ctd       = [0.8447, 0.8506, 0.8487, 0.8467]
ibs_d     = [0.3395, 0.3265, 0.2773, 0.2713]
ibs_death = [0.0805, 0.1241, 0.1972, 0.3194]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(rounds))
ax2 = ax.twinx()
l1, = ax.plot(x, ctd, marker='o', linewidth=2, color='#2e7d32', label='C_td dementia ↑')
l2, = ax2.plot(x, ibs_d, marker='s', linewidth=2, color='#d36b6b', label='IBS dementia ↓')
l3, = ax2.plot(x, ibs_death, marker='^', linewidth=2, linestyle='--', color='#5b8fb9', label='IBS death ↓')

for xi, v in enumerate(ctd):
    ax.annotate(f'{v:.4f}', (xi, v), textcoords='offset points', xytext=(0, -16),
                ha='center', fontsize=9, color='#2e7d32', fontweight='bold')
for xi, v in enumerate(ibs_d):
    ax2.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, -16),
                 ha='center', fontsize=9, color='#d36b6b')
for xi, v in enumerate(ibs_death):
    ax2.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, -16),
                 ha='center', fontsize=9, color='#5b8fb9')

ax.set_xticks(x); ax.set_xticklabels(rounds, fontsize=10)
ax.set_ylabel('C_td dementia (higher = better)', color='#2e7d32', fontsize=11)
ax2.set_ylabel('IBS (lower = better)', color='#d36b6b', fontsize=11)
ax.set_ylim(0.838, 0.856); ax2.set_ylim(0.05, 0.40)
ax.legend(handles=[l1, l2, l3], loc='upper right', fontsize=9)
ax.set_title('Self-Training Progression (no SST → SST 1% → 2% → 5%)\n'
             'SST 1% peaks for C_td dementia; higher SST trades for IBS dementia but hurts IBS death',
             fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = FIG_DIR / 'fig_sst_progression.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

print('\nAll figures saved to', FIG_DIR)
