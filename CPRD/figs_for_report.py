"""
Generate figures for the English progress report — clean models only.

Figures produced (PNG, dpi 150):
  1. fig_model_comparison.png — bar chart of clean models only
                                 (no pre-leakage experiments — those go to appendix)
  2. fig_trade_compare.png   — Our two best clean models vs TRADE (AUROC + PPV only)
  3. fig_code_overlap.png    — Stacked-bar visualisation of our HES code-set vs TRADE-strict
  4. fig_sst_progression.png — line plot across SST rounds (no SST → 1% → 2% → 5%),
                                 all labels positioned below lines to avoid overlap
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path('/Data0/swangek_data/991/CPRD/figs')
FIG_DIR.mkdir(exist_ok=True)

# =================================================================
# Fig 1: Clean-models comparison (5 trained, 1 in-progress placeholder)
# =================================================================
models = [
    'Single-GP\n+ HESstatic',
    'Dual-gated\n+ HESstatic',
    'Dual + HESstatic\n+ SST1%',
    'Dual + HESstatic\n+ SST2%',
    'Dual + HESstatic\n+ SST5%',
]
ctd_dem  = [0.8451, 0.8447, 0.8506, 0.8487, 0.8467]
auroc_5y = [0.9236, 0.9284, 0.9271, 0.9288, 0.9219]
ppv_1pct = [68.89, 71.11, 55.56, 62.22, 53.33]

x = np.arange(len(models))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={'hspace': 0.35})

# Color scheme: baseline (single) blue, base dual blue, SST greens
colors = ['#5b8fb9', '#5b8fb9', '#2e7d32', '#2e7d32', '#2e7d32']

# Panel 1: C_td dementia
ax1.bar(x, ctd_dem, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(ctd_dem):
    ax1.text(i, v + 0.0015, f'{v:.4f}', ha='center', fontsize=10)
ax1.set_ylim(0.840, 0.855)
ax1.set_ylabel('Cohort C_td (Antolini) — dementia', fontsize=11)
ax1.set_title('Cohort-Level C_td (Antolini), Dementia (canonical 8,241 test cohort, V2 clean labels)',
              fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels([])

# Panel 2: AUROC
ax2.bar(x, auroc_5y, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(auroc_5y):
    ax2.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10)
ax2.set_ylim(0.90, 0.94)
ax2.set_ylabel('5y AUROC', fontsize=11)
ax2.set_title('5y AUROC — exclude_early_censored policy', fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels([])

# Panel 3: PPV
ax3.bar(x, ppv_1pct, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(ppv_1pct):
    ax3.text(i, v + 1.0, f'{v:.1f}%', ha='center', fontsize=10)
ax3.set_ylim(0, 80)
ax3.set_ylabel('5y PPV @ top 1% (%)', fontsize=11)
ax3.set_title('5y PPV at top 1% — clinical screening yield', fontsize=11)
ax3.axhline(2.68, color='red', linestyle='--', linewidth=0.6, label='Cohort prevalence 2.68%')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=0, ha='center', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = FIG_DIR / 'fig_model_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# =================================================================
# Fig 2: TRADE comparison — AUROC + PPV only (no C-index since TRADE doesn't report)
# =================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios':[1, 2]})
w = 0.27

# Panel A: AUROC only
mA = ['5y AUROC']
oa1 = [0.9236]
oa2 = [0.9271]
ta  = [0.735]
xA = np.arange(len(mA))
ax_a.bar(xA - w, oa1, w, label='Single-GP + HESstatic\n(our baseline)', color='#5b8fb9', edgecolor='black', linewidth=0.5)
ax_a.bar(xA,     oa2, w, label='Dual + HESstatic + SST1%\n(our best)', color='#2e7d32', edgecolor='black', linewidth=0.5)
ax_a.bar(xA + w, ta,  w, label='TRADE (NYU)', color='#d36b6b', edgecolor='black', linewidth=0.5)
for v, xv in [(oa1[0], xA[0]-w), (oa2[0], xA[0]), (ta[0], xA[0]+w)]:
    ax_a.text(xv, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
ax_a.set_xticks(xA)
ax_a.set_xticklabels(mA)
ax_a.set_ylim(0, 1.05)
ax_a.set_title('Discrimination (higher = better)')
ax_a.legend(loc='lower right', fontsize=8.5)
ax_a.grid(axis='y', alpha=0.3)
ax_a.axhline(0.5, color='red', linestyle='--', linewidth=0.6, alpha=0.5)

# Panel B: PPV
mB = ['PPV @ top 1%', 'PPV @ top 5%', 'PPV @ top 10%']
ob1 = [68.89, 27.88, 19.03]
ob2 = [55.56, 29.65, 19.25]
tb  = [39.2,  27.8,  22.4]
xB = np.arange(len(mB))
ax_b.bar(xB - w, ob1, w, label='Single-GP + HESstatic', color='#5b8fb9', edgecolor='black', linewidth=0.5)
ax_b.bar(xB,     ob2, w, label='Dual + HESstatic + SST1%', color='#2e7d32', edgecolor='black', linewidth=0.5)
ax_b.bar(xB + w, tb,  w, label='TRADE', color='#d36b6b', edgecolor='black', linewidth=0.5)
for i, (v1, v2, vt) in enumerate(zip(ob1, ob2, tb)):
    ax_b.text(i - w, v1 + 1, f'{v1:.1f}%', ha='center', fontsize=9)
    ax_b.text(i,     v2 + 1, f'{v2:.1f}%', ha='center', fontsize=9)
    ax_b.text(i + w, vt + 1, f'{vt:.1f}%', ha='center', fontsize=9)
ax_b.set_xticks(xB)
ax_b.set_xticklabels(mB)
ax_b.set_ylim(0, max(ob1+ob2+tb)+10)
ax_b.set_title('PPV @ top X% (higher = better)')
ax_b.set_ylabel('PPV (%)')
ax_b.legend(loc='upper right', fontsize=9)
ax_b.grid(axis='y', alpha=0.3)
ax_b.axhline(2.68, color='blue', linestyle=':', linewidth=0.6, alpha=0.6, label='_Our 5y prevalence (2.68%)')
ax_b.axhline(7.2, color='red', linestyle=':', linewidth=0.6, alpha=0.6, label='_TRADE 5y prevalence (7.2%)')

plt.suptitle('Direct Comparison vs TRADE (no C-index — TRADE does not report it)', fontsize=12)
plt.tight_layout()
out = FIG_DIR / 'fig_trade_compare.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# =================================================================
# Fig 3: Code-set overlap — stacked-bar visualisation
# (Replaces broken Venn. Our set is a strict subset of TRADE-strict.)
# =================================================================
fig, ax = plt.subplots(figsize=(10, 4))

# Numbers
ours = 5996
both = 5996      # Ours ∩ TRADE-strict (= ours, since ours ⊂ TRADE)
trade_only = 174 # TRADE has but we don't (G31.0 + G31.8 + G31.1)
ours_only = 0    # We have but TRADE doesn't

# Two horizontal stacked bars: "Ours" and "TRADE-strict"
y_ours = 1
y_trade = 0

# Ours: 5996 (all in intersection)
ax.barh(y_ours, ours, left=0, color='#5b8fb9', edgecolor='black', linewidth=0.5, label='Ours ∩ TRADE-strict')
# TRADE-strict: 5996 (intersection) + 174 (TRADE only)
ax.barh(y_trade, both, left=0, color='#5b8fb9', edgecolor='black', linewidth=0.5)
ax.barh(y_trade, trade_only, left=both, color='#d36b6b', edgecolor='black', linewidth=0.5, label='TRADE-strict only (G31.0 + G31.8 + G31.1)')

# Labels on bars
ax.text(ours/2, y_ours, f'5,996  (4.50% of cohort)', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
ax.text(both/2, y_trade, f'5,996', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
ax.text(both + trade_only/2, y_trade, f'174', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax.set_yticks([y_ours, y_trade])
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

# Delete the broken Venn if exists
old_venn = FIG_DIR / 'fig_code_venn.png'
if old_venn.exists():
    old_venn.unlink()
    print(f"Removed {old_venn}")

# =================================================================
# Fig 4: SST progression — ALL labels BELOW lines (no overlap)
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

# All annotations BELOW the corresponding point (negative y offset)
for xi, v in enumerate(ctd):
    ax.annotate(f'{v:.4f}', (xi, v), textcoords='offset points',
                xytext=(0, -16), ha='center', fontsize=9, color='#2e7d32', fontweight='bold')
for xi, v in enumerate(ibs_d):
    ax2.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                 xytext=(0, -16), ha='center', fontsize=9, color='#d36b6b')
for xi, v in enumerate(ibs_death):
    ax2.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                 xytext=(0, -16), ha='center', fontsize=9, color='#5b8fb9')

ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=10)
ax.set_ylabel('C_td dementia (higher = better)', color='#2e7d32', fontsize=11)
ax2.set_ylabel('IBS (lower = better)', color='#d36b6b', fontsize=11)
ax.set_ylim(0.838, 0.856)
ax2.set_ylim(0.05, 0.40)
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
