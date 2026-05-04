# 模型改进计划 v2：扩展 Static Features + 辅助损失

> **创建日期**: 2026-04-27
> **最后更新**: 2026-05-02
> **前置知识**: 请先阅读 `CPRD/PROJECT_KNOWLEDGE.md` 了解项目全貌
> **当前最好结果 (纠正后)**: Dual-backbone v2 (GP + HES, gated fusion, 22-dim clean static) → Dementia C_td = **0.743** (baseline: 0.733, +0.010)
> **⚠️ 重要**: 之前报告的 hes_static v1 (0.836)、dual v1 (0.845)、hes_static v2 (0.875) 结果均因 **时间泄露 (temporal leakage)** 而无效。详见 Section 8。
> **目标**: 在当前 dual-backbone 基础上进一步提升性能

---

## 目录

1. [当前模型回顾与改进动机](#1-当前模型回顾与改进动机)
2. [改进一：扩展 HES Static Features](#2-改进一扩展-hes-static-features)
3. [改进二：辅助损失（Auxiliary Loss）](#3-改进二辅助损失auxiliary-loss)
4. [实验执行计划](#4-实验执行计划)
5. [文件修改清单](#5-文件修改清单)
6. [验证方案](#6-验证方案)
7. [风险与备选方案](#7-风险与备选方案)
8. [⚠️ 时间泄露发现与纠正](#8-️-时间泄露-temporal-leakage-发现与纠正)

---

## 1. 当前模型回顾与改进动机

### 1.1 当前 dual-backbone 架构

```
Patient i:
  GP序列 (Read v2, ≤512 tokens) + 35-dim static (27 base + 8 HES) 
    → [GP Backbone (pretrained, 108K vocab)] → h_gp (384-dim) ─┐
                                                                ├─ Gated Fusion → h_fused (384-dim) → 单个 ODESurvCR Head → 竞争风险预测
  HES序列 (ICD-10 3-char, ≤256 tokens) + 27-dim static (base only)          │
    → [HES Backbone (pretrained, 1.5K vocab)] → h_hes (384-dim) ─┘
```

**当前只有一个 Survival Head**，融合后的 h_fused 直接送入 ODESurvCompetingRiskLayer。

### 1.2 历史实验结果

> **⚠️ 注意**: 标记为 "⚠️ INVALID" 的结果因时间泄露 (temporal leakage) 而无效。详见 Section 8。

| 方法 | Dementia C_td | Death C_td | Overall C_td | 备注 |
|------|--------------|------------|-------------|------|
| hes_aug (baseline) | 0.733 | 0.944 | 0.858 | ✅ GP + HES labels only (无泄露) |
| hes_fusion v5 | 0.720 | 0.898 | 0.788 | ✅ FAILED, 序列融合 (无泄露) |
| hes_static v1 | ~~0.836~~ | ~~0.944~~ | ~~0.885~~ | ⚠️ INVALID — 时间泄露 |
| dual-backbone v1 | ~~0.845~~ | ~~0.949~~ | ~~0.891~~ | ⚠️ INVALID — 时间泄露 |
| **dual-backbone v2 (纠正后)** | **0.743** | **0.951** | **0.855** | ✅ **当前最好 (clean train+test)** |

### 1.3 改进动机

**改进一（扩展 Static Features）**: 当前只有 8 个 HES 共病特征（stroke, MI, heart_failure, diabetes, delirium, TBI + 2 个连续量）。扩展到 22 维覆盖更多已知风险因素（高血压、房颤、抑郁、帕金森等）。**注意**: 之前报告的 hes_static 从 0.733 → 0.836 (+0.103) 的提升是时间泄露造成的虚假结果。纠正后，22-dim dual-backbone 的提升为 0.733 → 0.743 (+0.010)，仍有正向效果但幅度远小于预期。

**改进二（辅助损失）**: 当前 GP backbone 的监督信号只通过 fusion 后的 h_fused 间接传递。添加一个 GP-only 的辅助 survival head，让 GP backbone 同时接受直接监督，可以学到更好的 GP 表示。这是一种多任务学习策略，在深度学习中被广泛验证有效。

### 1.4 两个改进的独立性

这两个改进**完全独立**，可以分别实施、分别验证，也可以组合使用：

| 实验 | 改进一（扩展 static） | 改进二（辅助损失） |
|------|:---:|:---:|
| Exp A: static_v2 only | ✅ | ❌ |
| Exp B: aux_loss only | ❌ | ✅ |
| Exp C: static_v2 + aux_loss | ✅ | ✅ |

建议按 A → B → C 顺序执行，以量化每个改进的独立贡献。

---

## 2. 改进一：扩展 HES Static Features

### 2.1 目标

将 HES static features 从 8 维扩展到 **22 维**（8 原有 + 11 新共病 + 3 新连续特征），覆盖更多与痴呆相关的临床风险因素。

### 2.2 新增共病特征设计

#### 原有 8 维（保持不变）

| # | Feature | Type | ICD-10 | 意义 |
|---|---------|------|--------|------|
| 0 | `HES_TOTAL_ADMISSIONS` | Continuous | — | 住院负担 |
| 1 | `HES_TOTAL_UNIQUE_DIAG` | Continuous | — | 诊断复杂度 |
| 2 | `HES_HAS_STROKE` | Binary | I60-I69 | 血管性痴呆风险 |
| 3 | `HES_HAS_MI` | Binary | I21-I22 | 心血管风险 |
| 4 | `HES_HAS_HEART_FAILURE` | Binary | I50 | 心血管风险 |
| 5 | `HES_HAS_DIABETES` | Binary | E10-E14 | 已知风险因素 |
| 6 | `HES_HAS_DELIRIUM` | Binary | F05 | 强预测因子 |
| 7 | `HES_HAS_TBI` | Binary | S06 | 已知风险因素 |

#### 新增 11 个共病特征

| # | Feature | Type | ICD-10 Prefix | 临床意义 |
|---|---------|------|---------------|---------|
| 8 | `HES_HAS_HYPERTENSION` | Binary | I10, I11, I12, I13, I14, I15 | 血管性痴呆核心风险因素，中年高血压与晚年痴呆强相关 |
| 9 | `HES_HAS_ATRIAL_FIBRILLATION` | Binary | I48 | 通过中风路径增加血管性痴呆风险 |
| 10 | `HES_HAS_CKD` | Binary | N18 | 慢性肾病加速认知退化 |
| 11 | `HES_HAS_DEPRESSION` | Binary | F32, F33 | 既是痴呆前驱症状也是独立风险因素 |
| 12 | `HES_HAS_PARKINSON` | Binary | G20 | 路易体痴呆强相关 |
| 13 | `HES_HAS_EPILEPSY` | Binary | G40, G41 | 与痴呆双向因果关系 |
| 14 | `HES_HAS_OBESITY` | Binary | E66 | 中年肥胖增加痴呆风险 |
| 15 | `HES_HAS_HYPERLIPIDEMIA` | Binary | E78 | 心血管风险链 |
| 16 | `HES_HAS_COPD` | Binary | J44 | 缺氧导致认知损伤 |
| 17 | `HES_HAS_ALCOHOL` | Binary | F10 | 酒精性痴呆 |
| 18 | `HES_HAS_SLEEP_DISORDER` | Binary | G47 | 睡眠呼吸暂停与痴呆相关 |

#### 新增 3 个连续特征

| # | Feature | Type | 计算方式 | 归一化 | 意义 |
|---|---------|------|---------|--------|------|
| 19 | `HES_MEAN_STAY_DAYS` | Continuous | mean(disdate - admidate) | log(1+days)/log(31), cap 1.0 | 住院严重程度 |
| 20 | `HES_EMERGENCY_RATIO` | Continuous | emergency_count / total_admissions | 直接 [0,1] | 急性病负担 |
| 21 | `HES_YEARS_SINCE_LAST_ADMISSION` | Continuous | (index_date - last_admidate).years | min(years/20, 1.0) | 近期健康状况（值越小=最近越频繁住院）。**注意**: 使用 index_date 而非 study_end |

### 2.3 ICD-10 编码映射（完整）

```python
# 新增 COMORBIDITY_PREFIXES（添加到现有字典中）
NEW_COMORBIDITY_PREFIXES = {
    "hypertension":         ["I10", "I11", "I12", "I13", "I14", "I15"],
    "atrial_fibrillation":  ["I48"],
    "ckd":                  ["N18"],
    "depression":           ["F32", "F33"],
    "parkinson":            ["G20"],
    "epilepsy":             ["G40", "G41"],
    "obesity":              ["E66"],
    "hyperlipidemia":       ["E78"],
    "copd":                 ["J44"],
    "alcohol":              ["F10"],
    "sleep_disorder":       ["G47"],
}
```

**注意**: `F10` (酒精相关) 和 `G47` (睡眠障碍) 不会与排除列表 (`F00, F01, F02, F03, G30` 痴呆码) 冲突。

### 2.4 需要修改的文件

#### 文件 1: `build_hes_summary_features.py`（主要修改）

**路径**: `CPRD/examples/modelling/SurvivEHR/build_hes_summary_features.py`

**修改内容**: 扩展特征提取逻辑

```python
"""
修改后的 build_hes_summary_features.py
======================================
从 8 维扩展到 22 维 HES summary features。

新增 11 个共病 + 3 个连续特征。
"""

import math
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"
OUTPUT_PATH = "/Data0/swangek_data/991/CPRD/data/hes_summary_features.pickle"

# ============================================================
# Feature definitions (22 features total)
# ============================================================
FEATURE_NAMES = [
    # --- 原有 8 维 ---
    "HES_TOTAL_ADMISSIONS",         # 0: continuous, log-scaled
    "HES_TOTAL_UNIQUE_DIAG",        # 1: continuous, log-scaled
    "HES_HAS_STROKE",               # 2: binary
    "HES_HAS_MI",                   # 3: binary
    "HES_HAS_HEART_FAILURE",        # 4: binary
    "HES_HAS_DIABETES",             # 5: binary
    "HES_HAS_DELIRIUM",             # 6: binary
    "HES_HAS_TBI",                  # 7: binary
    # --- 新增 11 个共病 ---
    "HES_HAS_HYPERTENSION",         # 8: binary
    "HES_HAS_ATRIAL_FIBRILLATION",  # 9: binary
    "HES_HAS_CKD",                  # 10: binary
    "HES_HAS_DEPRESSION",           # 11: binary
    "HES_HAS_PARKINSON",            # 12: binary
    "HES_HAS_EPILEPSY",             # 13: binary
    "HES_HAS_OBESITY",              # 14: binary
    "HES_HAS_HYPERLIPIDEMIA",       # 15: binary
    "HES_HAS_COPD",                 # 16: binary
    "HES_HAS_ALCOHOL",              # 17: binary
    "HES_HAS_SLEEP_DISORDER",       # 18: binary
    # --- 新增 3 个连续特征 ---
    "HES_MEAN_STAY_DAYS",           # 19: continuous, log-scaled
    "HES_EMERGENCY_RATIO",          # 20: continuous, [0,1]
    "HES_YEARS_SINCE_LAST_ADMISSION",  # 21: continuous, scaled
]
NUM_FEATURES = len(FEATURE_NAMES)  # 22

# Comorbidity ICD-10 prefix definitions (原有 + 新增)
COMORBIDITY_PREFIXES = {
    # 原有
    "stroke":               ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    "mi":                   ["I21", "I22"],
    "heart_failure":        ["I50"],
    "diabetes":             ["E10", "E11", "E12", "E13", "E14"],
    "delirium":             ["F05"],
    "tbi":                  ["S06"],
    # 新增
    "hypertension":         ["I10", "I11", "I12", "I13", "I14", "I15"],
    "atrial_fibrillation":  ["I48"],
    "ckd":                  ["N18"],
    "depression":           ["F32", "F33"],
    "parkinson":            ["G20"],
    "epilepsy":             ["G40", "G41"],
    "obesity":              ["E66"],
    "hyperlipidemia":       ["E78"],
    "copd":                 ["J44"],
    "alcohol":              ["F10"],
    "sleep_disorder":       ["G47"],
}

# 共病名→特征索引映射 (binary features 从 index 2 开始)
COMORBIDITY_INDEX = {
    "stroke": 2, "mi": 3, "heart_failure": 4, "diabetes": 5,
    "delirium": 6, "tbi": 7,
    "hypertension": 8, "atrial_fibrillation": 9, "ckd": 10,
    "depression": 11, "parkinson": 12, "epilepsy": 13,
    "obesity": 14, "hyperlipidemia": 15, "copd": 16,
    "alcohol": 17, "sleep_disorder": 18,
}

# Dementia codes to EXCLUDE (used for labels, not features)
DEMENTIA_PREFIXES = ["F00", "F01", "F02", "G30"]
DEMENTIA_EXACT = {"F03"}

STUDY_END_DATE = datetime(2022, 10, 31)


def _is_dementia(code: str) -> bool:
    for p in DEMENTIA_PREFIXES:
        if code.startswith(p):
            return True
    return code in DEMENTIA_EXACT


def main():
    print("=" * 60)
    print("  Building HES Summary Features v2 (22 dims)")
    print("=" * 60)

    # ============================================================
    # Step 1: 读取 hesin.csv — 提取入院信息
    # ============================================================
    print("\nStep 1: Reading hesin.csv for admission info...")
    hesin = pd.read_csv(
        HESIN_CSV,
        usecols=["dnx_hesin_id", "eid", "admidate", "disdate", "admimeth"],
        dtype={"dnx_hesin_id": str, "eid": str, "admimeth": str},
    )

    # 1a. 每患者入院次数 (原有 feature 0)
    admission_counts = hesin.groupby("eid")["dnx_hesin_id"].nunique()
    print(f"  {len(admission_counts)} patients with HES records")

    # 1b. 每患者住院天数 (新 feature 19: HES_MEAN_STAY_DAYS)
    hesin["admidate_dt"] = pd.to_datetime(hesin["admidate"], errors="coerce")
    hesin["disdate_dt"] = pd.to_datetime(hesin["disdate"], errors="coerce")
    hesin["stay_days"] = (hesin["disdate_dt"] - hesin["admidate_dt"]).dt.days
    hesin.loc[hesin["stay_days"] < 0, "stay_days"] = np.nan  # 清理异常值
    mean_stay = hesin.dropna(subset=["stay_days"]).groupby("eid")["stay_days"].mean()

    # 1c. 急诊入院比例 (新 feature 20: HES_EMERGENCY_RATIO)
    # admimeth: 21-28 = emergency admissions
    hesin["is_emergency"] = hesin["admimeth"].apply(
        lambda x: 1 if str(x)[:2] in ["21", "22", "23", "24", "25", "26", "27", "28"] else 0
    )
    emergency_counts = hesin.groupby("eid")["is_emergency"].sum()

    # 1d. 最近一次入院距研究结束的年数 (新 feature 21: HES_YEARS_SINCE_LAST_ADMISSION)
    last_admission = hesin.dropna(subset=["admidate_dt"]).groupby("eid")["admidate_dt"].max()

    # ============================================================
    # Step 2: 读取 hesin_diag.csv — 提取诊断共病
    # ============================================================
    print("\nStep 2: Processing diagnoses from hesin_diag.csv...")
    diag = pd.read_csv(
        HESIN_DIAG_CSV,
        usecols=["eid", "diag_icd10"],
        dtype={"eid": str, "diag_icd10": str},
    )
    diag = diag.dropna(subset=["diag_icd10"])
    diag["diag_icd10"] = diag["diag_icd10"].str.strip()
    print(f"  {len(diag)} diagnosis records with valid ICD-10 codes")

    # Per-patient: unique diagnoses + comorbidity flags
    patient_unique_diag = defaultdict(set)
    patient_comorbidities = defaultdict(lambda: {k: False for k in COMORBIDITY_PREFIXES})

    for eid, icd10 in zip(diag["eid"], diag["diag_icd10"]):
        if _is_dementia(icd10):
            continue
        patient_unique_diag[eid].add(icd10)
        for comorbidity, prefixes in COMORBIDITY_PREFIXES.items():
            if any(icd10.startswith(p) for p in prefixes):
                patient_comorbidities[eid][comorbidity] = True

    # ============================================================
    # Step 3: 构建 22 维特征向量
    # ============================================================
    print("\nStep 3: Building 22-dim feature vectors...")
    all_patients = set(admission_counts.index) | set(patient_unique_diag.keys())

    features = {}
    for eid in all_patients:
        pid = int(eid)
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)

        # --- 原有连续特征 ---
        n_admissions = admission_counts.get(eid, 0)
        n_unique_diag = len(patient_unique_diag.get(eid, set()))
        feat[0] = min(math.log1p(n_admissions) / math.log(51), 1.0)
        feat[1] = min(math.log1p(n_unique_diag) / math.log(101), 1.0)

        # --- 所有共病 (binary, index 2-18) ---
        comorb = patient_comorbidities.get(eid, {})
        for comorb_name, feat_idx in COMORBIDITY_INDEX.items():
            feat[feat_idx] = float(comorb.get(comorb_name, False))

        # --- 新增连续特征 ---
        # Feature 19: HES_MEAN_STAY_DAYS
        ms = mean_stay.get(eid, 0.0)
        feat[19] = min(math.log1p(ms) / math.log(31), 1.0)

        # Feature 20: HES_EMERGENCY_RATIO
        total_adm = admission_counts.get(eid, 0)
        emerg_count = emergency_counts.get(eid, 0)
        feat[20] = float(emerg_count / total_adm) if total_adm > 0 else 0.0

        # Feature 21: HES_YEARS_SINCE_LAST_ADMISSION
        last_adm = last_admission.get(eid, None)
        if last_adm is not None and pd.notna(last_adm):
            years_since = (STUDY_END_DATE - last_adm.to_pydatetime()).days / 365.25
            feat[21] = min(max(years_since, 0.0) / 20.0, 1.0)
        else:
            feat[21] = 1.0  # 无入院记录 → 距离最远

        features[pid] = feat

    # ============================================================
    # Step 4: 统计
    # ============================================================
    print(f"\n  Total patients: {len(features)}")
    all_feats = np.stack(list(features.values()))
    for i, name in enumerate(FEATURE_NAMES):
        col = all_feats[:, i]
        nonzero_frac = (col > 0).mean()
        print(f"  {name:40s}  mean={col.mean():.4f}  std={col.std():.4f}  nonzero={nonzero_frac:.3f}")

    # ============================================================
    # Step 5: 保存
    # ============================================================
    print(f"\nSaving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"features": features, "feature_names": FEATURE_NAMES}, f)
    print("Done.")


if __name__ == "__main__":
    main()
```

**关键修改点汇总**:
1. `FEATURE_NAMES` 从 8 扩展到 22
2. `COMORBIDITY_PREFIXES` 新增 11 个共病
3. `COMORBIDITY_INDEX` 映射共病名到特征索引
4. `hesin.csv` 多读 `disdate` 和 `admimeth` 列，用于计算住院天数和急诊比例
5. 特征构建部分新增 3 个连续特征的计算逻辑
6. 输出格式不变：`{patient_id: np.array([22 floats])}`

#### 文件 2: `build_dementia_cr_hes_static.py`（无需修改）

此文件不需要修改。它从 `hes_summary_features.pickle` 动态读取 `feature_names`，然后按名称添加列：

```python
# build_dementia_cr_hes_static.py:128-129 (现有逻辑)
for i, col_name in enumerate(feature_names):
    df[col_name] = hes_feat_matrix[:, i].astype(np.float32)
```

只要 `feature_names` 从 8 变成 22，它会自动添加 22 列。

#### 文件 3: `foundational_loader.py`（无需修改）

`_parquet_row_to_static_covariates()` 中的 HES 特征检测逻辑已经是通用的：

```python
# foundational_loader.py (现有逻辑)
hes_cols = sorted(col for col in row_df.index if col.startswith("HES_"))
for col in hes_cols:
    val = float(row_df.get(col, 0.0))
    covariates.append(np.asarray(val).reshape((1, -1)))
```

22 个 `HES_*` 列会被自动检测和追加。

#### 配置变更

`num_static_covariates` 需要从 35 更新为 **49**（27 base + 22 HES）。

以下配置文件需要修改：

| 配置文件 | 修改 |
|---------|------|
| `config_FineTune_Dementia_CR_hes_static.yaml` | `num_static_covariates: 49` |
| `config_FineTune_Dementia_CR_hes_static_eval.yaml` | `num_static_covariates: 49` |
| `config_FineTune_Dementia_CR_dual.yaml` | `num_static_covariates: 49` |
| `config_FineTune_Dementia_CR_dual_eval.yaml` | `num_static_covariates: 49` |

**注意**: 实际上 `num_static_covariates` 在 `run_experiment.py:64` / `run_dual_experiment.py:86` 中是动态从数据读取的：

```python
cfg.data.num_static_covariates = next(iter(dm.test_dataloader()))['static_covariates'].shape[1]
```

所以 YAML 中的值只是初始值，运行时会被覆盖。但为了文档清晰，建议同步更新 YAML。

#### Checkpoint 加载的 static_proj 兼容性

GP pretrain checkpoint 的 `static_proj` 是 `nn.Linear(27, 384)`。
- 之前 hes_static (35-dim) 已经处理了 partial load: 前 27 列从 checkpoint 加载，后 8 列零初始化
- 现在扩展到 49-dim: 同样的逻辑，前 27 列从 checkpoint，后 22 列零初始化
- `setup_finetune_experiment.py` 和 `setup_dual_finetune_experiment.py` 中的 partial load 逻辑已经是通用的：

```python
# setup_dual_finetune_experiment.py:270-277 (现有逻辑)
if 'static_proj' in k:
    new_p = model_sd[k].clone()
    if v.dim() == 2:
        new_p[:v.shape[0], :v.shape[1]] = v  # 只复制前 27 列
    else:
        new_p[:v.shape[0]] = v
    gp_mapping[k] = new_p
```

这段代码已经可以处理任意大小扩展，**无需修改**。

#### `dual_data_module.py` 中 HES static covariates 的处理

当前 `DualCollateWrapper` 取 GP static 的前 27 维作为 HES backbone 的 static：

```python
# dual_data_module.py:256 (现有逻辑)
hes_static = gp_batch['static_covariates'][:, :27].clone()
```

扩展到 49-dim 后，GP 的 static_covariates 变为 49-dim，但 HES backbone 仍然只用 27-dim。上面的 `[:, :27]` 切片逻辑**无需修改**。

### 2.5 执行步骤

```bash
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR

# Step 1: 修改 build_hes_summary_features.py (按 Section 2.4 的代码)
# Step 2: 重新生成特征
$PYTHON build_hes_summary_features.py
# 验证输出应包含 22 个特征名

# Step 3: 重新构建数据集 (会覆盖 FineTune_Dementia_CR_hes_static/)
$PYTHON build_dementia_cr_hes_static.py
# 验证: 读取任一 parquet 文件，确认有 22 个 HES_* 列

# Step 4: 更新配置中的 num_static_covariates (可选，运行时会自动读取)
# 修改 config_FineTune_Dementia_CR_hes_static.yaml: num_static_covariates: 49
# 修改 config_FineTune_Dementia_CR_dual.yaml: num_static_covariates: 49

# Step 5: 训练 hes_static_v2 (验证 static features 单独效果)
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/last.ckpt
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static \
    optim.accumulate_grad_batches=16

# Step 6: 评估 hes_static_v2
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static_eval

# Step 7: 训练 dual_v2 (在 static_v2 数据集上训练双模型)
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/last.ckpt
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual

# Step 8: 评估 dual_v2
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval
```

**重要**: Step 5-6 (hes_static_v2) 和 Step 7-8 (dual_v2) 使用的 checkpoint 文件名与现有的相同。如果需要保留现有 checkpoint，请先重命名：

```bash
# 备份现有 checkpoint
cd /Data0/swangek_data/991/CPRD/output/checkpoints
mv crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt crPreTrain_small_1337_FineTune_Dementia_CR_hes_static_v1.ckpt
mv crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt crPreTrain_small_1337_FineTune_Dementia_CR_dual_v1.ckpt
```

或者改 config 中的 `fine_tune_id` 为 `FineTune_Dementia_CR_hes_static_v2` 和 `FineTune_Dementia_CR_dual_v2`。

---

## 3. 改进二：辅助损失（Auxiliary Loss）

### 3.1 目标

在 dual-backbone 架构中，为 GP backbone 添加一个独立的辅助 survival head，让 GP backbone 同时接受直接的生存监督信号。

### 3.2 设计原理

当前架构的信号流：

```
当前 (无辅助损失):
  h_gp  ─→ fusion ─→ h_fused ─→ surv_layer ─→ loss_main
  h_hes ─↗

GP backbone 的梯度必须经过 fusion 层才能获得监督信号。
```

添加辅助损失后：

```
改进后 (有辅助损失):
  h_gp  ─→ fusion ─→ h_fused ─→ surv_layer_main ─→ loss_main  ─┐
  h_hes ─↗                                                       ├→ total_loss = loss_main + λ * loss_aux
  h_gp  ─→ surv_layer_aux ──→ loss_aux ─────────────────────────┘

GP backbone 同时获得:
  1. 间接信号 (通过 fusion → surv_layer_main)
  2. 直接信号 (通过 surv_layer_aux)
```

**好处**:
- GP backbone 获得更强的直接监督，不依赖 fusion 层的反向传播
- 辅助 head 起到正则化效果，防止 GP backbone 过度依赖 HES 信号
- 在测试时只用主 head 预测，不增加推理开销
- λ 控制辅助信号强度（推荐从 0.1 开始调）

### 3.3 需要修改的文件

#### 文件 1: `setup_dual_finetune_experiment.py`（核心修改）

**路径**: `CPRD/examples/modelling/SurvivEHR/setup_dual_finetune_experiment.py`

修改 `DualFineTuneExperiment` 类，添加辅助 survival head 和相应的 loss 计算。

**具体修改**:

##### 修改 1: `__init__` 中添加辅助 head

在现有 `self.surv_layer = ODESurvCompetingRiskLayer(...)` 之后添加：

```python
# 在 __init__ 中，self.surv_layer = ODESurvCompetingRiskLayer(...) 之后添加:

# Auxiliary GP-only survival head
self.aux_loss_weight = getattr(cfg, 'aux_loss_weight', 0.0)  # λ, 从配置读取
if self.aux_loss_weight > 0:
    self.aux_surv_layer = ODESurvCompetingRiskLayer(
        hidden_dim, [32, 32], num_risks=num_risks, device=desurv_device
    )
    logging.info(f"Auxiliary GP-only survival head enabled, weight={self.aux_loss_weight}")
else:
    self.aux_surv_layer = None
    logging.info("Auxiliary GP-only survival head disabled")
```

##### 修改 2: `forward()` 中计算辅助 loss

在现有的 `surv_dict, losses_desurv = self.surv_layer.predict(...)` 之后添加辅助 loss 计算。

**完整的 forward() 替换** — 从 `# Fusion` 注释开始到函数结束：

```python
    # Fusion
    h_fused = self.model.fusion(h_gp, h_hes)

    # === Main Survival prediction (与当前完全相同) ===
    in_hidden_state = torch.stack((h_fused, h_fused), dim=1)
    target_tokens = torch.hstack((
        torch.zeros((bsz, 1), device=self.device), target_token
    ))
    target_ages = torch.hstack((
        torch.zeros((bsz, 1), device=self.device), target_age_delta
    ))
    target_attention_mask = torch.ones_like(target_tokens) == 1

    # Sample weights
    _sample_weights = None
    if self.weighting_mode != "none" and return_loss:
        reduced_k = self.reduce_to_outcomes(target_token.reshape(-1))
        _sample_weights = compute_sample_weights(
            t=target_age_delta.reshape(-1),
            k=reduced_k,
            mode=self.weighting_mode,
            event_lambda=self.event_lambda,
            alpha=self.weight_alpha,
            tau=self.weight_tau,
            w_t_max=self.w_t_max,
            w_total_max=self.w_total_max,
        )

    surv_dict, losses_desurv = self.surv_layer.predict(
        in_hidden_state,
        target_tokens=self.reduce_to_outcomes(target_tokens),
        target_ages=target_ages,
        attention_mask=target_attention_mask,
        is_generation=is_generation,
        return_loss=return_loss,
        return_cdf=return_generation,
        sample_weights=_sample_weights,
    )

    if return_loss:
        loss_main = torch.sum(torch.stack(losses_desurv))
    else:
        loss_main = None

    # === Auxiliary GP-only loss (新增) ===
    loss_aux = torch.tensor(0.0, device=self.device)
    if return_loss and self.aux_surv_layer is not None and self.aux_loss_weight > 0:
        in_hidden_state_gp = torch.stack((h_gp, h_gp), dim=1)
        _, losses_aux = self.aux_surv_layer.predict(
            in_hidden_state_gp,
            target_tokens=self.reduce_to_outcomes(target_tokens),
            target_ages=target_ages,
            attention_mask=target_attention_mask,
            is_generation=False,
            return_loss=True,
            return_cdf=False,
            sample_weights=_sample_weights,
        )
        loss_aux = torch.sum(torch.stack(losses_aux))

    # === Total loss ===
    if return_loss:
        loss = loss_main + self.aux_loss_weight * loss_aux
    else:
        loss = None

    outputs = {"surv": surv_dict}
    losses = {
        "loss": loss,
        "loss_desurv": loss_main,
        "loss_aux": loss_aux if self.aux_surv_layer is not None else torch.tensor(0.0, device=self.device),
        "loss_values": torch.tensor(0.0, device=self.device),
    }

    return outputs, losses, h_fused
```

##### 修改 3: `configure_optimizers()` 中添加辅助 head 的参数组

```python
def configure_optimizers(self):
    params = [
        {"params": self.model.gp_transformer.parameters(),
         "lr": self.cfg.optim.learning_rate},         # 5e-5
        {"params": self.model.hes_transformer.parameters(),
         "lr": self.cfg.optim.learning_rate},         # 5e-5
        {"params": self.model.fusion.parameters(),
         "lr": self.cfg.fine_tuning.head.learning_rate},  # 5e-4
        {"params": self.surv_layer.parameters(),
         "lr": self.cfg.fine_tuning.head.learning_rate},  # 5e-4
    ]
    # 辅助 head 也用高学习率 (从零学习)
    if self.aux_surv_layer is not None:
        params.append({
            "params": self.aux_surv_layer.parameters(),
            "lr": self.cfg.fine_tuning.head.learning_rate,  # 5e-4
        })
    optimizer = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
    }
```

##### 修改 4: `training_step` / `validation_step` 中 log 辅助 loss

现有代码已经 log 所有 loss_dict 中的 key：

```python
def training_step(self, batch, batch_idx):
    _, loss_dict, _ = self(batch)
    for k, v in loss_dict.items():
        if v is not None:
            self.log(f"train_{k}", v, prog_bar=False, logger=True, sync_dist=True)
    return loss_dict['loss']
```

因为我们在 losses dict 中已经加了 `loss_aux`，所以 `train_loss_aux` 和 `val_loss_aux` 会自动被 log。**无需修改**。

#### 文件 2: 配置文件

在 YAML config 中添加 `aux_loss_weight` 参数。

**修改 `config_FineTune_Dementia_CR_dual.yaml`**:

在顶层添加：

```yaml
# Auxiliary loss configuration
aux_loss_weight: 0.1    # λ for GP-only auxiliary loss (0 = disabled)
```

**修改 `config_FineTune_Dementia_CR_dual_eval.yaml`**:

同样添加：

```yaml
aux_loss_weight: 0.1    # 必须与训练时一致（影响模型结构）
```

**注意**: `aux_loss_weight` 影响模型结构（是否创建 `aux_surv_layer`）。eval 时也需要设置相同的值，否则 `load_from_checkpoint` 会因参数不匹配而失败。

#### DDP 兼容性

如果 `aux_loss_weight: 0`，`aux_surv_layer` 为 None，不会有未使用参数问题。
如果 `aux_loss_weight > 0`，`aux_surv_layer` 参与 loss 计算，DDP 正常。
当前已有 `strategy = "ddp_find_unused_parameters_true"`，即使辅助 head 在 eval 阶段不参与（test_step 中 return_loss=True 但不需要 CDF），参数也不会报错。

### 3.4 关于 `aux_loss_weight` (λ) 的选择

| λ | 含义 | 推荐场景 |
|---|------|---------|
| 0.0 | 禁用辅助损失 | baseline / 对照实验 |
| 0.1 | 轻微辅助 | **推荐起点** |
| 0.3 | 中等辅助 | 如果 0.1 有效，可以试更大 |
| 0.5 | 强辅助 | 上限，再大可能干扰主 loss |
| 1.0 | 等权重 | 不推荐（辅助变成主要目标） |

建议先用 **λ=0.1** 跑一次完整训练+评估，然后视结果调整。

### 3.5 执行步骤

```bash
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR

# Step 1: 修改 setup_dual_finetune_experiment.py (按 Section 3.3 的代码)
# Step 2: 修改配置文件 (添加 aux_loss_weight)

# Step 3: 训练 (需要新的 fine_tune_id 以区分)
rm -f /Data0/swangek_data/991/CPRD/output/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual \
    experiment.fine_tune_id=FineTune_Dementia_CR_dual_auxloss \
    aux_loss_weight=0.1

# Step 4: 评估
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval \
    experiment.fine_tune_id=FineTune_Dementia_CR_dual_auxloss \
    aux_loss_weight=0.1
```

**提示**: 也可以创建独立的配置文件 `config_FineTune_Dementia_CR_dual_auxloss.yaml` 和 `config_FineTune_Dementia_CR_dual_auxloss_eval.yaml`，内容与现有 dual config 相同，仅添加 `aux_loss_weight: 0.1` 并修改 `fine_tune_id`。

---

## 4. 实验执行计划

### 4.1 推荐执行顺序

```
实验 A: hes_static_v2 (22-dim static, 单 backbone)
  ├─ 修改 build_hes_summary_features.py
  ├─ 重新生成特征 + 重新构建数据集
  ├─ 训练 + 评估
  └─ 对比: hes_static (8-dim) C_td=0.836 vs hes_static_v2 (22-dim) C_td=?

实验 B: dual_v2 (22-dim static, 双 backbone, 无辅助损失)
  ├─ 复用实验 A 的数据集
  ├─ 训练 + 评估
  └─ 对比: dual (8-dim) C_td=0.845 vs dual_v2 (22-dim) C_td=?

实验 C: dual_v2_auxloss (22-dim static, 双 backbone, 辅助损失 λ=0.1)
  ├─ 修改 setup_dual_finetune_experiment.py
  ├─ 复用实验 A 的数据集
  ├─ 训练 + 评估
  └─ 对比: dual_v2 C_td=? vs dual_v2_auxloss C_td=?
```

### 4.2 完整 pipeline 脚本

**新建文件**: `CPRD/run_improvement_v2_pipeline.sh`

```bash
#!/bin/bash
set -e

PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"

cd "$WORK_DIR"

echo "============================================================"
echo "  Improvement v2 Pipeline"
echo "  Start: $(date)"
echo "============================================================"

# ============================================================
# PHASE 1: Rebuild data with 22-dim static features
# ============================================================
echo "===== Phase 1: Rebuild data ====="

# Step 1.1: Generate 22-dim HES features
echo "--- Step 1.1: Generate HES summary features (22 dims) ---"
$PYTHON build_hes_summary_features.py

# Step 1.2: Rebuild hes_static dataset with new features
echo "--- Step 1.2: Rebuild hes_static dataset ---"
$PYTHON build_dementia_cr_hes_static.py

# ============================================================
# PHASE 2: Experiment A — hes_static_v2 (single backbone, 22-dim)
# ============================================================
echo "===== Phase 2: Experiment A — hes_static_v2 ====="

rm -f "${CKPT_DIR}/last.ckpt"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static \
    optim.accumulate_grad_batches=16

# Eval
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_hes_static_eval

echo "===== Exp A DONE ====="

# ============================================================
# PHASE 3: Experiment B — dual_v2 (dual backbone, 22-dim, no aux loss)
# ============================================================
echo "===== Phase 3: Experiment B — dual_v2 ====="

rm -f "${CKPT_DIR}/last.ckpt"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual

# Eval
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval

echo "===== Exp B DONE ====="

# ============================================================
# PHASE 4: Experiment C — dual_v2_auxloss (dual backbone, 22-dim, aux loss λ=0.1)
# ============================================================
echo "===== Phase 4: Experiment C — dual_v2_auxloss ====="

rm -f "${CKPT_DIR}/last.ckpt"
# 注意：使用不同的 fine_tune_id 以避免覆盖 Exp B 的 checkpoint

# Train
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual \
    experiment.fine_tune_id=FineTune_Dementia_CR_dual_auxloss \
    experiment.notes="Dual backbone + 22-dim static + aux loss lambda=0.1" \
    aux_loss_weight=0.1

# Eval
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval \
    experiment.fine_tune_id=FineTune_Dementia_CR_dual_auxloss \
    aux_loss_weight=0.1

echo "===== Exp C DONE ====="

echo "============================================================"
echo "  All experiments complete: $(date)"
echo "============================================================"
```

### 4.3 预计时间

| 阶段 | 预计时间 |
|------|---------|
| Phase 1: 数据重建 | ~30 分钟 |
| Phase 2: Exp A (hes_static_v2 train + eval) | ~25 小时 + 10 分钟 |
| Phase 3: Exp B (dual_v2 train + eval) | ~44 小时 + 10 分钟 |
| Phase 4: Exp C (dual_v2_auxloss train + eval) | ~44 小时 + 10 分钟 |
| **总计** | **~5 天** |

如果时间有限，可以只做 Phase 1 + Phase 2（验证 static features 效果），再决定是否继续 Phase 3-4。

---

## 5. 文件修改清单

### 5.1 改进一（扩展 Static Features）

| # | 文件 | 操作 | 复杂度 | 说明 |
|---|------|------|--------|------|
| 1 | `build_hes_summary_features.py` | **修改** | 中 | 核心修改：扩展到 22 维 |
| 2 | `config_FineTune_Dementia_CR_hes_static.yaml` | 修改 | 低 | `num_static_covariates: 49` (可选) |
| 3 | `config_FineTune_Dementia_CR_hes_static_eval.yaml` | 修改 | 低 | 同上 |
| 4 | `config_FineTune_Dementia_CR_dual.yaml` | 修改 | 低 | 同上 |
| 5 | `config_FineTune_Dementia_CR_dual_eval.yaml` | 修改 | 低 | 同上 |

**不需要修改的文件**: `build_dementia_cr_hes_static.py`, `foundational_loader.py`, `setup_finetune_experiment.py`, `setup_dual_finetune_experiment.py`, `dual_data_module.py`, `run_dual_experiment.py`。全部兼容。

### 5.2 改进二（辅助损失）

| # | 文件 | 操作 | 复杂度 | 说明 |
|---|------|------|--------|------|
| 6 | `setup_dual_finetune_experiment.py` | **修改** | 中 | 添加 aux_surv_layer + aux loss |
| 7 | `config_FineTune_Dementia_CR_dual.yaml` | 修改 | 低 | 添加 `aux_loss_weight: 0.1` |
| 8 | `config_FineTune_Dementia_CR_dual_eval.yaml` | 修改 | 低 | 添加 `aux_loss_weight: 0.1` |

**不需要修改的文件**: `dual_backbone.py`, `dual_data_module.py`, `run_dual_experiment.py`, `competing_risk.py`。

### 5.3 总结

| 文件 | 改进一 | 改进二 |
|------|:---:|:---:|
| `build_hes_summary_features.py` | ✏️ 修改 | — |
| `setup_dual_finetune_experiment.py` | — | ✏️ 修改 |
| `config_*_dual.yaml` (train) | ✏️ 可选 | ✏️ 修改 |
| `config_*_dual_eval.yaml` | ✏️ 可选 | ✏️ 修改 |
| `config_*_hes_static.yaml` | ✏️ 可选 | — |
| `config_*_hes_static_eval.yaml` | ✏️ 可选 | — |

**零风险**: 改进一只改数据构建脚本，不影响模型代码。改进二通过 `aux_loss_weight: 0` 默认值向后兼容，不影响现有模型行为。

---

## 6. 验证方案

### 6.1 改进一验证

| 检查项 | 预期 |
|--------|------|
| `hes_summary_features.pickle` 中 `feature_names` 长度 | 22 |
| 每患者特征向量维度 | 22 |
| Parquet 文件中 `HES_*` 列数 | 22 |
| 模型 `num_static_covariates` | 49 (27 + 22) |
| GP backbone `static_proj` weight shape | (384, 49) |
| HES backbone `static_proj` weight shape | (384, 27) — 不变 |
| 新增共病 nonzero 比例合理 | 每个特征 >0 的比例应在 0.01-0.30 之间 |
| hes_static_v2 Dementia C_td | ~~≥ 0.836~~ 原目标无效；纠正后 baseline=0.733 |

### 6.2 改进二验证

| 检查项 | 预期 |
|--------|------|
| 日志显示 "Auxiliary GP-only survival head enabled" | 是 |
| WandB 中出现 `train_loss_aux` 和 `val_loss_aux` | 是 |
| `loss_aux` 正常下降（不为零不为 NaN） | 是 |
| `loss_main` + 0.1 * `loss_aux` ≈ `loss` | 是 |
| DDP 无 unused parameters 报错 | 是 |
| 参数量增加约 0.5M（辅助 head） | 是 |
| dual_v2_auxloss Dementia C_td | ≥ 0.743（纠正后的 dual v2 baseline） |

### 6.3 总结果对比表

> **⚠️ 重要更新 (2026-05-02)**: 下表中标记为 "⚠️ INVALID" 的结果因 HES 特征时间泄露而无效。详见 Section 8。

| 实验 | Static dims | Aux loss | Dementia C_td | Death C_td | Overall C_td | 与 baseline 对比 | 状态 |
|------|:-----------:|:--------:|:-------------:|:----------:|:------------:|:--------------:|:----:|
| hes_aug (baseline) | — | — | 0.733 | 0.944 | 0.858 | baseline | ✅ 有效 |
| hes_static v1 (LEAKY) | 8 | — | ~~0.836~~ | ~~0.944~~ | ~~0.885~~ | ~~+0.103~~ | ⚠️ **INVALID** |
| dual v1 (LEAKY) | 8 | ❌ | ~~0.845~~ | ~~0.949~~ | ~~0.891~~ | ~~+0.112~~ | ⚠️ **INVALID** |
| Exp A: hes_static_v2 (LEAKY) | 22 | — | ~~0.875~~ | ~~0.961~~ | ~~0.915~~ | ~~+0.142~~ | ⚠️ **INVALID (2026-04-28)** |
| Exp A 验证: leaky model + clean test | 22 | — | 0.706 | — | — | -0.027 | ❌ 低于 baseline |
| **Exp B: dual_v2 (CLEAN)** | **22** | ❌ | **0.743** | **0.951** | **0.855** | **+0.010** | ✅ **有效 (2026-05-01)** |
| Exp C: dual_v2_auxloss | 22 | ✅ λ=0.1 | — | — | — | — | ⏳ 待执行 |

**Exp A 事后分析**: 之前报告的 C_td=0.875 完全是时间泄露造成的虚假提升。当使用正确过滤的测试数据评估该模型时，C_td 降至 0.706，低于 baseline (0.733)。这说明模型学到的是泄露信息的 shortcut，而非真实的预测模式。

**Exp B 分析 (纠正后)**: 使用正确时间过滤的 22-dim HES 特征重新训练和测试的 dual-backbone 模型，Dementia C_td 为 0.743，比 baseline 0.733 提升 +0.010。虽然远不如之前虚假报告的 +0.142，但这是真实的、无泄露的改进。

---

## 7. 风险与备选方案

### 7.1 改进一的风险

**风险**: 新增特征可能带来噪声或多重共线性（如高血压和心衰高度相关）。

**缓解**:
- 所有新特征都基于已知的临床证据，不是随机选择
- 如果 22-dim 效果不如 8-dim，可以通过 feature selection 找到最优子集
- Linear projection (`static_proj`) 本身有一定的特征选择能力（零权重=忽略特征）

**备选**: 如果 22-dim 不好，可以尝试：
- 选择 8 + 3-5 个最重要的新增共病（如高血压、房颤、抑郁）
- 只加连续特征（住院天数、急诊比例），不加更多共病

### 7.2 改进二的风险

**风险 1**: 辅助 loss 权重 λ 不合适，可能干扰主 loss 的优化。

**缓解**: 默认 λ=0.1 很保守。如果效果不好，先试 λ=0.05 或 λ=0.3。

**风险 2**: 辅助 head 增加显存占用（ODESurvCompetingRiskLayer 包含 ODE solver）。

**缓解**: 辅助 head 参数量很小（~0.5M），显存增加可忽略。如果 OOM，减小 batch_size 到 12 或增加 accumulate_grad_batches 到 48。

**风险 3**: `load_from_checkpoint` 需要 `aux_loss_weight` 匹配。

**缓解**: eval config 中必须设置与训练相同的 `aux_loss_weight`。如果忘记，会报 checkpoint key mismatch 错误——错误信息明确，容易排查。

### 7.3 后续可探索的方向（不在本计划范围内）

如果这两个改进效果好，后续可以考虑：

1. **Cross-Attention Fusion**: 让 GP 的 last token 对 HES 的所有 token 做 attention（比 gated fusion 更细粒度）
2. **HES level≤2**: 重新 pretrain HES backbone 时包含副诊断，丰富 HES 序列
3. **λ 调度**: aux_loss_weight 随训练进行而衰减（前期强辅助，后期减弱让主 loss 主导）
4. **Ensemble**: hes_static + dual 两个模型的 CDF 加权平均

---

## 8. ⚠️ 时间泄露 (Temporal Leakage) 发现与纠正

> **发现日期**: 2026-04-29
> **纠正完成日期**: 2026-05-01
> **影响范围**: 所有使用 HES static features 的实验 (hes_static v1, v2, dual v1)

### 8.1 问题描述

`build_hes_summary_features.py` 原始版本在构建 HES summary features 时，**没有按照患者的 index date (72岁) 进行时间过滤**。这意味着：

- 一个在 72 岁时被预测的患者，其特征向量中包含了 72 岁之后的住院和诊断记录
- 模型可以"看到未来"——例如，如果一个患者在 75 岁时被诊断为谵妄 (F05)，该信息被编码在 index date (72岁) 时的特征中
- 这在生存分析中构成严重的时间泄露

### 8.2 泄露的规模

| 数据 | 过滤前 (全部) | 过滤后 (仅 index date 之前) | 被过滤掉的比例 |
|------|:----------:|:-------------------:|:---------:|
| 入院记录 (hesin.csv) | 4,238,372 | 2,990,537 (70.6%) | **29.4%** |
| 诊断记录 (hesin_diag.csv) | 17,421,258 | 10,419,676 (59.8%) | **40.2%** |
| 谵妄 (F05) 患病率 | 2.0% | ~0.5% | **75% 的谵妄诊断发生在 index date 之后** |

### 8.3 泄露如何影响模型

1. **训练 + 测试都有泄露 → 虚假高性能**: 模型在训练时学到"谵妄 = 更可能发生痴呆"的 shortcut（因为许多谵妄诊断实际上发生在痴呆之后或同期），测试时同样的未来信息可用，metrics 被严重虚高。

2. **泄露训练 + 正确测试 → 低于 baseline**: 当用正确过滤的测试数据评估泄露训练的模型时：
   - hes_static v2 (leaky train): Dementia C_td = **0.706** (vs baseline 0.733, **-0.027**)
   - dual v1 (leaky train): Dementia C_td = **0.704** (vs baseline 0.733, **-0.029**)
   - 模型学到的 shortcut 在正确数据上不可用，导致性能反而下降

3. **正确训练 + 正确测试 → 小幅真实提升**:
   - dual v2 (clean train + clean test): Dementia C_td = **0.743** (vs baseline 0.733, **+0.010**)

### 8.4 修复方法

在 `build_hes_summary_features.py` 中添加时间过滤逻辑：

```python
# 1. 从 GP 数据库加载患者出生年份，计算 index date
yob_lookup = load_year_of_birth_lookup()  # 新函数
index_dates = {str(pid): yob + pd.DateOffset(years=72) for pid, yob in yob_lookup.items()}

# 2. 过滤入院记录：只保留 admidate < index_date 的记录
hesin["index_date"] = hesin["eid"].map(index_dates)
hesin = hesin[hesin["admidate_dt"] < hesin["index_date"]]

# 3. 过滤诊断记录：只保留来自已过滤入院记录的诊断
pre_index_admissions = set(zip(hesin["eid"], hesin["dnx_hesin_id"]))
diag_filtered = diag[
    diag.apply(lambda r: (r["eid"], r["dnx_hesin_id"]) in pre_index_admissions, axis=1)
]

# 4. Feature 21 改为相对于 index_date 计算（原来使用固定的 STUDY_END_DATE）
years_since = (idx_date - last_adm).days / 365.25
```

### 8.5 纠正后的完整实验流程

```
2026-04-29: 发现时间泄露
  ├─ 修复 build_hes_summary_features.py（添加时间过滤）
  ├─ 重新生成 hes_summary_features.pickle（clean 22-dim）
  ├─ 重新构建 FineTune_Dementia_CR_hes_static 数据集（clean）
  ├─ 验证: leaky hes_static_v2 + clean test → C_td=0.706 (低于baseline)
  ├─ 验证: leaky dual_v1 + clean test → C_td=0.704 (低于baseline)
  └─ 确认泄露问题

2026-04-30~05-01: 使用正确数据重新训练
  ├─ 重新训练 dual-backbone v2（22-dim clean static, clean train）
  ├─ 评估: clean dual v2 + clean test → C_td=0.743 (+0.010 vs baseline)
  └─ 确认真实改进幅度

结论: 真实提升 +0.010，而非之前虚假的 +0.103 ~ +0.142
```

### 8.6 教训总结

1. **在生存分析中，所有特征必须只使用 index date 之前的信息** — 这是最基本的原则但极易在实践中被忽略
2. **异常大的提升应引起怀疑** — 8 个简单的 binary 共病特征不应该带来 +0.103 的 C_td 提升
3. **验证方法: leaky train + clean test** — 如果性能低于 baseline，说明模型学到了泄露的 shortcut
4. **HES 的真实价值是有限的** — 纠正后的提升仅 +0.010，说明 pre-index HES 信息与 GP 记录存在较大重叠

---

## 附录 A: 关键文件路径参考

| 文件 | 路径 | 用途 |
|------|------|------|
| HES 特征提取 | `CPRD/examples/modelling/SurvivEHR/build_hes_summary_features.py` | 生成 22-dim 特征 |
| 数据集构建 | `CPRD/examples/modelling/SurvivEHR/build_dementia_cr_hes_static.py` | 构建 Parquet 数据集 |
| Dual experiment | `CPRD/examples/modelling/SurvivEHR/setup_dual_finetune_experiment.py` | 添加辅助 loss |
| Dual model | `CPRD/src/models/survival/task_heads/dual_backbone.py` | 不修改 |
| Dual data | `CPRD/examples/modelling/SurvivEHR/dual_data_module.py` | 不修改 |
| Dual entry point | `CPRD/examples/modelling/SurvivEHR/run_dual_experiment.py` | 不修改 |
| Survival head | `CPRD/src/modules/head_layers/survival/competing_risk.py` | 不修改，辅助 head 复用 |
| DeSurv ODE | `CPRD/src/modules/head_layers/survival/desurv.py` | 不修改 |
| GP Pretrain config | `CPRD/examples/modelling/SurvivEHR/confs/config_CompetingRisk11M.yaml` | 参考 |
| Dual train config | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual.yaml` | 修改 |
| Dual eval config | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual_eval.yaml` | 修改 |
| Static train config | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_hes_static.yaml` | 修改 (可选) |
| Static eval config | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_hes_static_eval.yaml` | 修改 (可选) |
| GP pretrained ckpt | `CPRD/output/checkpoints/crPreTrain_small_1337.ckpt` | 不修改 |
| HES pretrained ckpt | `CPRD/output/checkpoints/crPreTrain_HES_1337.ckpt` | 不修改 |
| HES features pickle | `CPRD/data/hes_summary_features.pickle` | 重新生成 |
| HES static dataset | `CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/` | 重新生成 |

## 附录 B: 关键代码行号参考

| 文件 | 行号 | 内容 | 改进相关性 |
|------|------|------|-----------|
| `build_hes_summary_features.py:34-43` | `FEATURE_NAMES` 列表 | **改进一修改点** |
| `build_hes_summary_features.py:47-54` | `COMORBIDITY_PREFIXES` 字典 | **改进一修改点** |
| `build_hes_summary_features.py:109-133` | 特征构建循环 | **改进一修改点** |
| `build_dementia_cr_hes_static.py:128-129` | HES 列写入 Parquet | 改进一自动兼容 |
| `foundational_loader.py:662-738` | `_parquet_row_to_static_covariates()` | 改进一自动兼容 |
| `setup_dual_finetune_experiment.py:99-101` | `self.surv_layer = ODESurvCR(...)` | **改进二修改点: 在此后添加 aux_surv_layer** |
| `setup_dual_finetune_experiment.py:159-206` | `forward()` — fusion + loss | **改进二修改点: 添加辅助 loss** |
| `setup_dual_finetune_experiment.py:229-250` | `configure_optimizers()` | **改进二修改点: 添加 aux 参数组** |
| `setup_dual_finetune_experiment.py:270-277` | GP static_proj partial load | 改进一自动兼容 |
| `dual_data_module.py:256` | `hes_static = gp_batch[:, :27]` | 改进一自动兼容 |
| `run_dual_experiment.py:86` | `num_static_covariates` 动态读取 | 改进一自动兼容 |
