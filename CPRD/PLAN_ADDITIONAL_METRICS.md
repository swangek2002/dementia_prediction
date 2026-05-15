# Plan: Compute Additional Evaluation Metrics for V3 Model

## Background

**请先阅读项目架构文件**: `CPRD/PROJECT_KNOWLEDGE.md`，特别是 Section 6.7 (V3 Self-Training), Section 7.4 (Dual-Backbone Architecture), 和 Section 6.3 (Temporal Leakage) 了解项目全貌。

我们的最佳模型 (V3 self-training, Dementia C_td=0.7685) 目前只输出三个指标: C_td, IBS, INBLL。需要额外跑 AUROC@5y、Top-K precision、Harrell's C 等指标，以便和文献中其他模型对比。

**模型输出机制**: DeSurv (Neural ODE) 竞争风险模型，输出每个患者的两条 CIF (Cumulative Incidence Function) 曲线：
- `CIF_dementia(t)` = P(dementia before time t | patient history)
- `CIF_death(t)` = P(death before time t | patient history)

每条曲线在 `t_eval = np.linspace(0, 1, 1000)` 上采样（1000个时间点），`supervised_time_scale = 5.0`，所以模型时间 t=1.0 对应真实 5 年。

## 目标指标

| 指标 | 用途 | 对比文献 |
|------|------|---------|
| **AUROC@5y dementia** | 5年dementia判别力 | UKBDRS (0.80), Wang (0.810), Botz (0.776) |
| **AUROC@5y death** | 5年死亡判别力 | Gu (0.866) |
| **Top-K precision@5y (1%, 5%, 10%)** | 高风险人群精度 | DemRisk (37%@1%), NYU EHR-BERT (39.2%@1%) |
| **Harrell's C (cause-specific)** | 整体排序一致性 | Yuan (0.749) |
| **Calibration data@5y** | 校准度 | 可视化用 |

## 重要约束

1. **时间范围**: 模型CIF只覆盖 0-5 年，**无法计算 10y AUROC**（需重训才行）
2. **测试集统计** (8,257 patients):
   - 376 dementia, 451 death, 7,430 censored
   - 5年内dementia: 230例, 5年内death: 217例
   - ≥5年随访: 1,982 patients（足够算AUROC@5y）
   - ≥10年随访: 仅49人（即使能算也不够）
3. **不修改任何已有代码**，只新建脚本

## 执行步骤

### Step 1: 创建 test set inference 脚本

**创建文件**: `CPRD/examples/modelling/SurvivEHR/inference_test_metrics.py`

**参考**: 已有的 `inference_train_cif.py`（同目录），但需要做以下改动：

1. **改 checkpoint 路径** → V3 最佳模型:
   ```
   CKPT_PATH = "/Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v3.ckpt"
   ```

2. **改 dataset 路径** → V3 test split:
   ```
   GP_DS_PATH = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static_v3/"
   ```

3. **使用 test set** 而不是 train set:
   ```python
   # 使用 dm.test_set 而不是 dm.train_set
   patient_index = build_patient_index(dm.test_set)
   test_loader = DataLoader(dataset=dm.test_set, ..., shuffle=False)
   ```
   注意: FoundationalDataModule 需要在构造时指定 `split` 或在后续访问 `dm.test_set`。看 `inference_train_cif.py` 中如何做的——它构造 dm 后直接用 `dm.train_set`。同理你用 `dm.test_set`。

4. **保存两个 risk 的完整 CIF 曲线** + event time from index:
   
   inference_train_cif.py 只保存 dementia CIF 在 event time 处的值。这里需要保存更多信息:

   ```python
   # 从 parquet 中额外提取: index_age (=72), prediction_age (倒数第二个event的age)
   # 这样可以算 event time from index = last_age - 72

   # 保存内容 (每个患者一行):
   # - patient_id, label (dementia/death/censored)
   # - event_time_from_index_years: (last_age - 72.0) 年, 从index date起的真实时间
   # - event_time_scaled: target_age_delta (模型空间的时间)
   # - cif_dementia_1y, cif_dementia_2y, cif_dementia_3y, cif_dementia_5y: CIF_dementia 在 1/2/3/5 年处的值
   # - cif_death_1y, cif_death_2y, cif_death_3y, cif_death_5y: CIF_death 在 1/2/3/5 年处的值
   # - cif_dementia_at_event, cif_death_at_event: CIF 在实际 event time 处的值
   ```

   **关键: 如何提取 CIF 在特定年份处的值:**
   ```python
   t_eval = experiment.surv_layer.t_eval  # np.linspace(0, 1, 1000)
   
   # 5年 = t_model=1.0, 对应 t_eval 的最后一个点 (index 999)
   # 3年 = t_model=0.6, 对应 t_eval 中最接近 0.6 的点
   # 2年 = t_model=0.4
   # 1年 = t_model=0.2
   
   for year in [1, 2, 3, 5]:
       t_target = year / SUPERVISED_TIME_SCALE  # e.g., 5/5.0 = 1.0
       t_idx = min(np.argmin(np.abs(t_eval - t_target)), len(t_eval) - 1)
       cif_dem_at_year = dementia_cdf[i, t_idx]
       cif_death_at_year = death_cdf[i, t_idx]
   ```

   **关键: 提取两个 risk 的 CIF:**
   ```python
   all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)
   pred_cdfs = all_outputs["surv"]["surv_CDF"]
   dementia_cdf = pred_cdfs[0]  # shape (bsz, 1000) — risk 0 = dementia (31 codes)
   death_cdf = pred_cdfs[1]     # shape (bsz, 1000) — risk 1 = death (1 code)
   ```

   **关键: 如何获取 event time from index date:**
   
   `build_patient_index()` 函数需要改造，从 parquet 的 `DAYS_SINCE_BIRTH` 列提取:
   ```python
   last_age = row['DAYS_SINCE_BIRTH'][-1] / 365.25
   index_age = 72.0
   event_time_from_index = last_age - index_age  # 从 index date 起的真实年数
   ```

5. **输出 CSV**:
   ```
   OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v3.csv"
   ```

**运行命令**:
```bash
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD" \
/Data0/swangek_data/conda_envs/survivehr/bin/python inference_test_metrics.py
```

预计耗时: ~15分钟 (test set 8,257 patients，比 train set 的 119K 快很多)

### Step 2: 创建指标计算脚本

**创建文件**: `CPRD/examples/modelling/SurvivEHR/compute_additional_metrics.py`

这个脚本不需要 GPU，纯 pandas/numpy/sklearn 计算，读取 Step 1 输出的 CSV。

**需要计算的指标:**

#### 2.1 AUROC@5y (dementia)

```python
from sklearn.metrics import roc_auc_score

df = pd.read_csv("test_cif_v3.csv")

# 定义 5y binary label:
#   positive = dementia 且 event_time_from_index <= 5
#   negative = 在 5y 时仍无事件 (event_time_from_index > 5, 即 censored/death/dementia 都在5年之后)
#   排除: 在 5y 内 death 或 censored 的患者 (outcome unknown at 5y)
#
# 竞争风险 AUROC 定义 (Cumulative-Dynamic):
#   cases = dementia before 5y
#   controls = event-free at 5y (alive and no dementia at t=5y)

t_star = 5.0
cases = (df['label'] == 'dementia') & (df['event_time_from_index_years'] <= t_star)
controls = df['event_time_from_index_years'] > t_star  # still event-free at 5y
# Note: death before 5y 也排除 (competing risk, 在5y时已不在risk set中)
# Note: censored before 5y 也排除 (不知道5y时的状态)

mask = cases | controls
y_true = cases[mask].astype(int).values
y_score = df.loc[mask, 'cif_dementia_5y'].values

auroc_dem_5y = roc_auc_score(y_true, y_score)
```

#### 2.2 AUROC@5y (death)

同理，但用 `cif_death_5y` 作为 risk score:
```python
cases_death = (df['label'] == 'death') & (df['event_time_from_index_years'] <= t_star)
controls_death = df['event_time_from_index_years'] > t_star
mask_death = cases_death | controls_death

y_true_death = cases_death[mask_death].astype(int).values
y_score_death = df.loc[mask_death, 'cif_death_5y'].values

auroc_death_5y = roc_auc_score(y_true_death, y_score_death)
```

#### 2.3 Top-K Precision@5y

```python
# 所有人按 cif_dementia_5y 降序排列
df_sorted = df.sort_values('cif_dementia_5y', ascending=False)

for pct in [0.01, 0.05, 0.10]:
    k = int(len(df_sorted) * pct)
    top_k = df_sorted.head(k)
    # 在 top-K 中，有多少在 5y 内真的得了 dementia
    true_positives = ((top_k['label'] == 'dementia') & (top_k['event_time_from_index_years'] <= 5.0)).sum()
    precision = true_positives / k
    print(f"Top {pct*100:.0f}% precision@5y: {precision:.3f} ({true_positives}/{k})")
```

#### 2.4 Harrell's C (cause-specific)

```python
from lifelines.utils import concordance_index

# Cause-specific for dementia:
#   Only consider patients who had dementia OR are censored (remove competing death events)
#   Or use the full population with cause-specific approach:
#   event_observed = True only if dementia occurred

mask_harrell = df['label'].isin(['dementia', 'censored'])
df_harrell = df[mask_harrell]

# Higher CIF = higher risk → we need to negate for concordance_index which expects "lower = higher risk" for T
# Actually lifelines concordance_index: C = P(risk_score_i > risk_score_j | T_i < T_j)
# So we use cif_dementia_5y as predicted risk (higher = more likely)
c_index = concordance_index(
    event_times=df_harrell['event_time_from_index_years'],
    predicted_scores=df_harrell['cif_dementia_5y'],  # higher = higher risk
    event_observed=(df_harrell['label'] == 'dementia')
)
print(f"Harrell's C (dementia, cause-specific): {c_index:.4f}")
```

注意: `lifelines` 已安装在 conda 环境中。如果没有，用 `sksurv`:
```python
from sksurv.metrics import concordance_index_censored
c_stat, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    event_indicator=(df_harrell['label'] == 'dementia').values,
    event_time=df_harrell['event_time_from_index_years'].values,
    estimate=df_harrell['cif_dementia_5y'].values,
)
```

#### 2.5 Calibration Data@5y

```python
# 将预测的 CIF@5y 分成 10 个 bin，对比每个 bin 内的实际 5y dementia 发生率
# 只对有 >=5y 随访或在 5y 内发生事件的患者计算

mask_cal = (df['event_time_from_index_years'] > 5.0) | \
           ((df['label'] == 'dementia') & (df['event_time_from_index_years'] <= 5.0))

df_cal = df[mask_cal].copy()
df_cal['actual_5y'] = ((df_cal['label'] == 'dementia') & (df_cal['event_time_from_index_years'] <= 5.0)).astype(int)
df_cal['pred_bin'] = pd.qcut(df_cal['cif_dementia_5y'], q=10, duplicates='drop')

calibration = df_cal.groupby('pred_bin').agg(
    mean_predicted=('cif_dementia_5y', 'mean'),
    mean_actual=('actual_5y', 'mean'),
    count=('actual_5y', 'count')
).reset_index()

print("\nCalibration@5y (10 bins):")
print(calibration[['mean_predicted', 'mean_actual', 'count']].to_string())
```

#### 2.6 也跑 1y, 2y, 3y 的 AUROC (bonus)

和 5y 一样，换用 `cif_dementia_1y`, `cif_dementia_2y`, `cif_dementia_3y`。

**运行命令**:
```bash
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD" \
/Data0/swangek_data/conda_envs/survivehr/bin/python compute_additional_metrics.py
```

### Step 3: 输出格式

脚本应打印清晰的汇总表:

```
=== V3 Model Additional Evaluation Metrics ===

AUROC:
  Dementia @1y: X.XXXX
  Dementia @2y: X.XXXX
  Dementia @3y: X.XXXX
  Dementia @5y: X.XXXX  (cf. UKBDRS=0.80, Wang=0.810, Botz=0.776)
  Death @5y:    X.XXXX  (cf. Gu=0.866)

Top-K Precision @5y:
  Top 1%:  XX.X% (N/M)  (cf. DemRisk=37%, NYU EHR-BERT=39.2%)
  Top 5%:  XX.X% (N/M)
  Top 10%: XX.X% (N/M)

Harrell's C (cause-specific):
  Dementia: X.XXXX  (cf. Yuan=0.749)
  Death:    X.XXXX

Calibration @5y:
  [table]

Already-reported metrics (from eval):
  C_td dementia: 0.7685
  C_td death:    0.9518
  IBS dementia:  0.1740
  IBS death:     0.1009
  INBLL dementia: 0.5101
  INBLL death:   0.3319
```

## 环境信息

```bash
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
```

## 关键文件参考

| 文件 | 用途 |
|------|------|
| `CPRD/PROJECT_KNOWLEDGE.md` | **项目架构和实验记录，必读** |
| `CPRD/examples/modelling/SurvivEHR/inference_train_cif.py` | **inference脚本模板，Step 1 参考这个写** |
| `CPRD/src/modules/head_layers/survival/competing_risk.py` | CIF 预测逻辑 (`_predict_cdf`, `t_eval`) |
| `CPRD/src/models/survival/custom_callbacks/clinical_prediction_model.py` | 现有指标计算逻辑 (C_td, IBS, INBLL via pycox EvalSurv) |
| `CPRD/examples/modelling/SurvivEHR/setup_dual_finetune_experiment.py` | DualFineTuneExperiment 的 forward 逻辑 |
| `CPRD/examples/modelling/SurvivEHR/dual_data_module.py` | DualCollateWrapper, HES tokenizer, HES cache |
| `FastEHR/FastEHR/dataloader/foundational_loader.py` | FoundationalDataModule, `convert_to_supervised()` |
| `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual_v3_eval.yaml` | V3 eval config (可参考参数) |

## 注意事项

1. **不要修改任何已有文件**。只创建新脚本。
2. `inference_train_cif.py` 中的 `build_patient_index()` 函数从 parquet 中读 `EVENT` 列判断 label，从 `DAYS_SINCE_BIRTH` 列可以算出 age。直接照搬并扩展即可。
3. `DualFineTuneExperiment.load_from_checkpoint(CKPT_PATH)` 加载模型，**不需要传额外参数**。
4. `pred_cdfs = all_outputs["surv"]["surv_CDF"]` 返回一个 list，`pred_cdfs[0]` = dementia CIF (shape `(bsz, 1000)`)，`pred_cdfs[1]` = death CIF (shape `(bsz, 1000)`)。
5. **CIF 值域是 [0, ~0.3]**（不是 [0,1]），因为这是 cumulative incidence（竞争风险下单个 cause 的概率不会到1）。
6. 模型时间 t ∈ [0,1] 对应真实 0-5 年。`cif_dementia_5y = dementia_cdf[i, 999]` (最后一个时间点)。
7. `event_time_from_index = DAYS_SINCE_BIRTH[-1] / 365.25 - 72.0` — 从 index date (age 72) 起算的真实年数。
8. **关于 AUROC 的可比性**: 我们的模型使用全纵向 GP 历史（包括 index date 后的 GP 就诊记录），而 UKBDRS 等只用 baseline 评估。所以我们有信息优势。在最终对比时需注明 "dynamic prediction using full longitudinal EHR" vs "baseline-only prediction"。
9. **lifelines 或 sksurv**: Harrell's C 用 `lifelines.utils.concordance_index` 或 `sksurv.metrics.concordance_index_censored`。先 `try import lifelines`，如果没有再用 sksurv。如果都没有，`pip install lifelines` 到 conda 环境。
10. 先跑 Step 1 (需要 GPU)，再跑 Step 2 (不需要 GPU)。

---

## Step 3 (新增 — 必须做): 时间对齐 CIF 修复

### 为什么要做这一步

Step 2 跑出来的结果有严重问题：
- AUROC@5y dementia = **0.5756** (远低于 C_td=0.7685，互相矛盾)
- Top 1% precision @5y = **3.7%** (远低于 DemRisk=37%)
- AUROC@5y death = 0.8462 (这个 OK)

**根本原因**：**时间参考系错位** (不是 CIF 饱和、不是模型本身差)

- 模型的 CIF@5y 含义：从**预测点（倒数第二个 event 时刻）**起算 5 年内的 dementia 概率
- AUROC@5y 的 label 含义：从**index date (age 72)** 起 5 年内是否得 dementia
- **两个时间窗口对不上！**

举例：某患者预测点在 age 75.2（即 δ_i = 3.2y）
- `cif_dementia_5y = 0.7566` 实际表示 "75.2→80.2 岁之间发病的概率"
- 但 AUROC label 问的是 "72→77 岁之间是否发病"
- 两个窗口只重叠 1.8 年，评估的根本不是同一件事

### 已确认的关键数据 (用现有 test_cif_v3.csv 算出)

```
δ_i (years from index to prediction point) 分布:
  median=1.95y, p25=0.88y, p75=3.60y, min=-1.13y, max=10.80y

Cohort sizes:
  δ_i ∈ [0, 5]: 7,575 patients (230 dementia cases @5y, 1,300 controls @5y)
  δ_i > 5: 682 patients (全是 controls，prediction 点已超过 age 77)
  δ_i < 1: 没有 ≥5y 随访的 controls (数据架构限制 → 排除"限制 cohort"路径)
```

→ 唯一可行方案：**per-patient time-aligned CIF**
   对每个患者，risk score 用 `τ_i = (5 - δ_i) / 5.0` 处的 CIF（而不是固定的 t=1.0）

### 3.1 改 inference 脚本

**新建** `inference_test_metrics_v2.py`（不修改 v1），主要差异：

#### 改动 A: 保存细粒度 CIF 网格

当前只存 1y/2y/3y/5y 4 个固定点，不够用。改为存 ~26 个点（每 0.2y 一个）：

```python
# t_eval = np.linspace(0, 1, 1000)
# 每 0.2y = 40 个 grid point
fine_grid_indices = list(range(0, 1000, 40)) + [999]  # ~26 indices
fine_grid_years = np.array(fine_grid_indices) / 999.0 * 5.0  # 真实年数 [0, 0.2, ..., 5.0]

# 在 batch loop 内:
for i in range(bsz):
    cif_dem_fine = dementia_cdf[i, fine_grid_indices]  # shape (~26,)
    cif_death_fine = death_cdf[i, fine_grid_indices]
    # 保存到一个全局 numpy 数组
    all_cif_dem[global_idx + i] = cif_dem_fine
    all_cif_death[global_idx + i] = cif_death_fine
```

最后存为 npz：

```python
np.savez("/Data0/swangek_data/991/CPRD/data/test_cif_v3_fine.npz",
         patient_ids=np.array([r["patient_id"] for r in results]),
         cif_dementia=all_cif_dem,    # shape (n_patients, ~26)
         cif_death=all_cif_death,
         t_grid_years=fine_grid_years)
```

#### 改动 B: 保存 π 值 (asymptotic marginal)

`_predict_cdf` 在 `competing_risk.py` 里同时返回 `preds` 和 `pis`，但 v1 脚本只用了 preds。这次也保存 pi：

```python
all_outputs, _, _ = experiment(batch_device, return_loss=False, return_generation=True)
pred_cdfs = all_outputs["surv"]["surv_CDF"]   # list of CIF arrays
pred_pis  = all_outputs["surv"]["surv_pi"]    # list of pi arrays (新增)

dementia_cdf = pred_cdfs[0]   # (bsz, 1000)
death_cdf    = pred_cdfs[1]
dementia_pi  = pred_pis[0]    # (bsz, 1000) — pi 跨 t 重复, 取 [:,-1] 即可
death_pi     = pred_pis[1]

# 在 results.append 里:
"pi_dementia": float(dementia_pi[i, -1]),   # asymptotic π
"pi_death":    float(death_pi[i, -1]),
```

#### 改动 C: 输出文件名

```python
OUTPUT_CSV = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_aligned.csv"  # 不覆盖 v1
OUTPUT_NPZ = "/Data0/swangek_data/991/CPRD/data/test_cif_v3_fine.npz"
```

CSV 中除了原有列，新增 `pi_dementia`, `pi_death` 两列。

**运行**:
```bash
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD" \
/Data0/swangek_data/conda_envs/survivehr/bin/python inference_test_metrics_v2.py
```
预计 ~15 分钟。

### 3.2 改指标脚本

**新建** `compute_metrics_aligned.py`（不修改 v1，让对比一目了然）。

核心代码：

```python
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score

df = pd.read_csv("/Data0/swangek_data/991/CPRD/data/test_cif_v3_aligned.csv")
npz = np.load("/Data0/swangek_data/991/CPRD/data/test_cif_v3_fine.npz")
t_grid_years = npz['t_grid_years']  # [0, 0.2, ..., 5.0]
cif_dem_grid = npz['cif_dementia']  # (n, ~26)
cif_death_grid = npz['cif_death']

# Step A: 计算 delta_i 和 tau_i (in real years)
df['delta_i'] = df['event_time_from_index_years'] - df['event_time_scaled'] * 5.0
df['tau_i_years'] = 5.0 - df['delta_i']

# 注意: npz 的 patient_ids 顺序应该和 df 一致 (inference 时 shuffle=False),
# 但建议显式按 patient_id merge 防止顺序错乱
assert (npz['patient_ids'] == df['patient_id'].values).all(), "顺序不一致, 需要 merge"

# Step B: per-patient 插值得到 time-aligned CIF
mask_valid = (df['delta_i'] >= 0) & (df['delta_i'] <= 5.0)
print(f"Valid cohort (delta_i in [0,5]): {mask_valid.sum()}/{len(df)}")
print(f"  Excluded delta_i<0: {(df['delta_i']<0).sum()}, delta_i>5: {(df['delta_i']>5).sum()}")

cif_dem_aligned = np.full(len(df), np.nan)
cif_death_aligned = np.full(len(df), np.nan)

for idx in df[mask_valid].index:
    t_target = df.loc[idx, 'tau_i_years']  # in years
    f_dem = interp1d(t_grid_years, cif_dem_grid[idx], kind='linear',
                     bounds_error=False, fill_value=(cif_dem_grid[idx, 0], cif_dem_grid[idx, -1]))
    f_death = interp1d(t_grid_years, cif_death_grid[idx], kind='linear',
                       bounds_error=False, fill_value=(cif_death_grid[idx, 0], cif_death_grid[idx, -1]))
    cif_dem_aligned[idx] = float(f_dem(t_target))
    cif_death_aligned[idx] = float(f_death(t_target))

df['cif_dementia_aligned_5y'] = cif_dem_aligned
df['cif_death_aligned_5y'] = cif_death_aligned

# Step C: AUROC@5y dementia (限制到 valid cohort, 三种 risk score 对比)
df_valid = df[mask_valid]
cases = (df_valid['label'] == 'dementia') & (df_valid['event_time_from_index_years'] <= 5.0)
controls = df_valid['event_time_from_index_years'] > 5.0
mask = cases | controls
y_true = cases[mask].astype(int).values

print(f"\nAUROC@5y dementia (cases={int(y_true.sum())}, controls={int((~y_true.astype(bool)).sum())}):")
for name, col in [
    ("CIF@5y (saturated, original)",  "cif_dementia_5y"),
    ("pi_dementia (asymptotic)",      "pi_dementia"),
    ("CIF time-aligned (the fix)",    "cif_dementia_aligned_5y"),
]:
    s = df_valid.loc[mask, col].values
    if np.isnan(s).any():
        print(f"  {name:40s} = SKIP (has NaN)")
        continue
    print(f"  {name:40s} = {roc_auc_score(y_true, s):.4f}")

# Step D: AUROC@5y death (同上, 用 death 的三种 score)
cases_d = (df_valid['label'] == 'death') & (df_valid['event_time_from_index_years'] <= 5.0)
controls_d = df_valid['event_time_from_index_years'] > 5.0
mask_d = cases_d | controls_d
y_true_d = cases_d[mask_d].astype(int).values
print(f"\nAUROC@5y death (cases={int(y_true_d.sum())}, controls={int((~y_true_d.astype(bool)).sum())}):")
for name, col in [
    ("CIF@5y (saturated, original)",  "cif_death_5y"),
    ("pi_death (asymptotic)",         "pi_death"),
    ("CIF time-aligned (the fix)",    "cif_death_aligned_5y"),
]:
    s = df_valid.loc[mask_d, col].values
    if np.isnan(s).any():
        continue
    print(f"  {name:40s} = {roc_auc_score(y_true_d, s):.4f}")

# Step E: Top-K precision @5y (用 time-aligned CIF)
print("\nTop-K precision @5y dementia (using time-aligned CIF):")
df_sorted = df_valid.sort_values('cif_dementia_aligned_5y', ascending=False)
for pct in [0.01, 0.05, 0.10]:
    k = int(len(df_sorted) * pct)
    top_k = df_sorted.head(k)
    tp = ((top_k['label']=='dementia') & (top_k['event_time_from_index_years']<=5.0)).sum()
    print(f"  Top {pct*100:>2.0f}%: {tp/k*100:5.1f}% ({tp}/{k})")

# Step F: Harrell's C (cause-specific) 用 pi_dementia 当 risk score (asymptotic, 不依赖时间)
from lifelines.utils import concordance_index
mask_h = df['label'].isin(['dementia', 'censored'])
dh = df[mask_h]
print(f"\nHarrell's C (cause-specific dementia, pi_dementia as score, n={len(dh)}):")
print(f"  {concordance_index(event_times=dh['event_time_from_index_years'], predicted_scores=dh['pi_dementia'], event_observed=(dh['label']=='dementia')):.4f}")
```

### 3.3 期望输出格式

```
=== Time-Aligned AUROC@5y Comparison (Step 3) ===

Valid cohort (delta_i in [0, 5]): 7575/8257 patients
  Excluded delta_i<0: N1, delta_i>5: 682

AUROC@5y Dementia (cases=230, controls=1300):
  CIF@5y (saturated, original)             = 0.5756 (sanity check vs Step 2 result)
  pi_dementia (asymptotic)                 = 0.XXXX (should be similar to CIF@5y)
  CIF time-aligned (the fix)               = 0.XXXX (THE KEY NUMBER)

AUROC@5y Death (cases=XXX, controls=XXXX):
  CIF@5y (original)                        = 0.8462
  pi_death (asymptotic)                    = 0.XXXX
  CIF time-aligned                         = 0.XXXX

Top-K Precision @5y Dementia (time-aligned):
  Top  1%: XX.X%  (cf. DemRisk=37%, NYU EHR-BERT=39.2%)
  Top  5%: XX.X%
  Top 10%: XX.X%

Harrell's C (cause-specific dementia, pi_dementia score): 0.XXXX
  (cf. Yuan=0.749)
```

### 3.4 三种结果情形的解读

| 情况 | aligned AUROC | π AUROC | 解读 | 论文怎么写 |
|------|--------------|---------|------|------------|
| A | 0.78+ | ~0.58 | ✅ 时间对齐是关键修复 | 报 aligned 数字 + 说明 dynamic prediction |
| B | 0.65~0.75 | ~0.58 | ⚠️ 对齐有帮助但不够 | 报 aligned + 讨论长 horizon 限制 |
| C | ~0.58 | ~0.58 | 🚫 模型在 5y horizon 上确实弱 | 强调 C_td 和 1-2y AUROC 才是模型强项 |

最可能是 **情况 A**（基于 C_td=0.7685 已经显示模型 discrimination 本身没问题）。

### 3.5 注意事项

1. **不要修改 v1 脚本**: 新建 `inference_test_metrics_v2.py` 和 `compute_metrics_aligned.py`
2. **不要覆盖 v1 输出文件**: 新文件用 `_aligned` / `_fine` 后缀
3. **保留 Step 2 的报告输出**，Step 3 的结果**追加**在后面，方便看对比
4. **PROJECT_KNOWLEDGE.md Section 6.7 必读**（V3 模型架构和数据流）
5. **关于 AUROC@5y 可比性**: 我们的模型是 **dynamic prediction**（用全纵向历史），UKBDRS 等是 **baseline-only**。即使时间对齐，本质上仍不完全 apples-to-apples，但 aligned CIF 已经是能给出的最严格的 5y AUROC 数字
6. **如果 npz 的 patient_ids 顺序和 csv 不一致**：用 patient_id 做 key merge，不要假设顺序对齐
7. **inference 时 `shuffle=False`**（v1 脚本里已经是这样）确保 patient index 和 dataloader 顺序一致
