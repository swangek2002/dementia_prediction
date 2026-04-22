# 双模型架构实现计划 (Dual-Backbone Architecture)

> **创建日期**: 2026-04-21
> **前置知识**: 请先阅读 `CPRD/PROJECT_KNOWLEDGE.md` 了解项目全貌
> **目标**: 将当前单 GP backbone 架构升级为 GP backbone + HES backbone 的双模型架构，通过 late fusion 融合两个独立 backbone 的 hidden states，再输入 survival head 进行预测

---

## 目录

1. [动机与设计原理](#1-动机与设计原理)
2. [总体架构图](#2-总体架构图)
3. [阶段一：HES Pretrain](#3-阶段一hes-pretrain)
4. [阶段二：Dual-Backbone Fine-tune](#4-阶段二dual-backbone-fine-tune)
5. [具体修改文件清单](#5-具体修改文件清单)
6. [验证方案](#6-验证方案)
7. [风险与备选方案](#7-风险与备选方案)

---

## 1. 动机与设计原理

### 1.1 为什么需要双模型

当前 best model (hes_static, C_td=0.836) 仅通过 8 维 HES summary statistics 就获得了 +0.103 的提升。但 HES 中还有大量未利用的信息：

- **17.5M** 条诊断记录（12,315 种 ICD-10 码）
- **4.2M** 条入院记录（含入院方式、住院天数、科室等）
- 每个患者的住院事件**时序模式**（如反复住院、住院频率变化等）

hes_static 用 8 个数字压缩了所有这些信息。直接序列融合 (fusion v5) 失败的原因是**模态冲突**——HES 事件插入 GP 序列导致截断和 backbone 学习的时序模式被破坏。

**双模型架构的核心思想**：给 GP 和 HES 各自一个独立的 transformer backbone，各自在自己的模态上预训练，各自编码各自的序列，最后在 hidden state 层面融合。这样：

1. GP backbone 的输入序列不受影响（没有截断问题）
2. HES backbone 在 HES 数据上专门学习住院时序模式
3. Fusion 层在 fine-tune 阶段学习如何组合两种信息

### 1.2 架构对比

```
当前架构 (hes_static):
  GP序列 → [GP Backbone] → hidden_state (384-dim) → survival_head
  HES信息 → 8维static → static_proj → 加到token embedding上

双模型架构:
  GP序列 → [GP Backbone (pretrained)]  → h_gp  (384-dim) ─┐
                                                            ├─ Fusion → h_fused → survival_head
  HES序列 → [HES Backbone (pretrained)] → h_hes (384-dim) ─┘
```

---

## 2. 总体架构图

### 2.1 数据流

```
Patient i:
  ├─ GP data:  诊断、处方、测量事件（Read v2码）  → GP序列 (≤512 tokens)
  │            + static covariates (27-dim)
  │
  └─ HES data: 住院诊断事件（ICD-10码）           → HES序列 (≤256 tokens)
               + static covariates (27-dim)       (可以复用GP的static)

GP路径:
  GP序列 → GP DataEmbeddingLayer(vocab_gp, 384, static=27)
         → GP TransformerBlocks(x6)
         → GP LayerNorm
         → h_gp[last_token] (384-dim)

HES路径:
  HES序列 → HES DataEmbeddingLayer(vocab_hes, 384, static=27)
          → HES TransformerBlocks(x6)
          → HES LayerNorm
          → h_hes[last_token] (384-dim)

Fusion:
  [h_gp; h_hes] (768-dim) → FusionLayer → h_fused (384-dim)
  h_fused → ODESurvCompetingRiskLayer → 生存预测

FusionLayer 选项:
  (A) 简单: Linear(768, 384) + ReLU
  (B) 带门控: GatedFusion: gate = σ(W_g · [h_gp; h_hes])
              h_fused = gate * W_gp(h_gp) + (1-gate) * W_hes(h_hes)
  (C) Cross-attention: h_gp attend to h_hes (更复杂，建议先用A/B)
```

### 2.2 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| HES 编码系统 | 使用原始 ICD-10 码（不翻译为 Read v2） | 避免翻译损失，HES backbone 直接学习 ICD-10 语义 |
| HES block_size | 256 | HES 序列较短（住院次数远少于 GP 就诊次数） |
| HES pretrain 方式 | 自监督 next-event prediction（与 GP pretrain 相同范式） | 保持一致性 |
| Fusion 位置 | Fine-tune 阶段（不在 pretrain） | 两个 backbone 独立预训练，fusion 层从零学习 |
| 无 HES 记录的患者 | h_hes 设为零向量 | Fusion 层可以学到当 h_hes≈0 时依赖 h_gp |
| 是否保留 hes_static | 是，GP backbone 继续用 35-dim static covariates | 保留已验证的有效信号 |

---

## 3. 阶段一：HES Pretrain

### 3.1 目标

创建一个独立的 HES transformer backbone，在 HES 住院数据上进行自监督预训练（next-event prediction），使其学会编码住院时序模式。

### 3.2 Step 1: 构建 HES SQLite 数据库

**新建文件**: `CPRD/examples/modelling/SurvivEHR/build_hes_database.py`

从 `hesin.csv` + `hesin_diag.csv` 构建一个**纯 HES** 的 SQLite 数据库，格式与 GP 数据库完全相同（`diagnosis_table` + `static_table`），以便复用 FoundationalDataModule。

#### 3.2.1 HES 数据库 schema

```sql
-- diagnosis_table: 完全复用GP格式
CREATE TABLE diagnosis_table (
    PRACTICE_ID integer,    -- 复用GP的PRACTICE_ID（用于保持相同的train/val/test分割）
    PATIENT_ID integer,     -- eid（与GP PATIENT_ID一致）
    EVENT text,             -- ICD-10 码（如 "I21", "E11", "F05" 等）
    DATE text               -- 诊断日期（格式 "YYYY-MM-DD"）
);

-- static_table: 复制GP的static_table（SEX, IMD, ETHNICITY, YEAR_OF_BIRTH等）
-- 这样HES backbone也能使用相同的27维static covariates
CREATE TABLE static_table (
    PRACTICE_ID integer,
    PATIENT_ID integer,
    ETHNICITY text,
    YEAR_OF_BIRTH text,
    SEX text,
    COUNTRY text,
    IMD text,
    HEALTH_AUTH text,
    INDEX_DATE text,
    START_DATE text,
    END_DATE text
);
```

#### 3.2.2 构建逻辑

```python
"""
build_hes_database.py
=====================
Build a standalone HES-only SQLite database for HES backbone pretraining.
Uses the same schema as the GP database so FoundationalDataModule can load it directly.

数据来源:
  - hesin.csv: 入院记录 (eid, admidate, disdate, ...)
  - hesin_diag.csv: 诊断记录 (eid, diag_icd10, level, ...)
  - GP DB static_table: 复用人口学信息

HES事件构建规则:
  1. 对每个入院记录 (hesin row), 获取该次入院的所有诊断 (hesin_diag)
  2. 每个 ICD-10 诊断码作为一个 EVENT, 日期为该次入院的 admidate
  3. ICD-10 码截断到 3 位 (如 "I219" → "I21") 以减少词汇量
  4. 仅保留 level=1 (主诊断) 的记录
     ——如果想要更丰富的信息，可以改为保留 level≤2 (主+副)
  5. PRACTICE_ID: 从GP static_table中查找该patient的practice，保持分割一致

输出: /Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db
"""

import os
import sqlite3
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

GP_DB = "/Data0/swangek_data/991/CPRD/data/example_exercise_database.db"
HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HESIN_CSV = "/Data0/swangek_data/991/CPRD/data/hesin.csv"
HESIN_DIAG_CSV = "/Data0/swangek_data/991/CPRD/data/hesin_diag.csv"

# ICD-10 截断位数 (3位 = 大类, 4位 = 含修饰符)
ICD10_TRUNCATE_LEN = 3

def main():
    # 1. 从 GP DB 复制 static_table
    if os.path.exists(HES_DB):
        os.remove(HES_DB)

    conn_gp = sqlite3.connect(GP_DB)
    conn_hes = sqlite3.connect(HES_DB)
    cur_hes = conn_hes.cursor()

    # 复制 static_table schema 和数据
    static_df = pd.read_sql("SELECT * FROM static_table", conn_gp)
    static_df.to_sql("static_table", conn_hes, if_exists="replace", index=False)
    print(f"Copied static_table: {len(static_df)} rows")

    # 建立 PATIENT_ID -> PRACTICE_ID 的映射
    pid_to_practice = dict(zip(static_df["PATIENT_ID"].astype(int),
                               static_df["PRACTICE_ID"].astype(int)))
    conn_gp.close()

    # 2. 创建空的 diagnosis_table
    cur_hes.execute("""
        CREATE TABLE diagnosis_table (
            PRACTICE_ID integer,
            PATIENT_ID integer,
            EVENT text,
            DATE text
        )
    """)

    # 3. 读取 hesin + hesin_diag, 构建 HES 事件
    print("Reading hesin.csv...")
    hesin = pd.read_csv(HESIN_CSV, usecols=["dnx_hesin_id", "eid", "admidate"])
    hesin = hesin.dropna(subset=["admidate"])
    hesin["eid"] = hesin["eid"].astype(int)
    # 建立 hesin_id -> (eid, admidate) 映射
    hesin_lookup = dict(zip(hesin["dnx_hesin_id"],
                            zip(hesin["eid"], hesin["admidate"])))

    print("Reading hesin_diag.csv...")
    hesin_diag = pd.read_csv(HESIN_DIAG_CSV,
                              usecols=["dnx_hesin_id", "eid", "diag_icd10", "level"])
    hesin_diag = hesin_diag.dropna(subset=["diag_icd10"])
    hesin_diag["eid"] = hesin_diag["eid"].astype(int)

    # 只保留主诊断 (level=1), 或者也可以选择 level<=2
    hesin_diag = hesin_diag[hesin_diag["level"] == 1]

    # ICD-10 截断
    hesin_diag["EVENT"] = hesin_diag["diag_icd10"].str[:ICD10_TRUNCATE_LEN]

    # 合并日期
    hesin_diag = hesin_diag.merge(
        hesin[["dnx_hesin_id", "admidate"]],
        on="dnx_hesin_id",
        how="left"
    )
    hesin_diag = hesin_diag.dropna(subset=["admidate"])

    # 添加 PRACTICE_ID
    hesin_diag["PRACTICE_ID"] = hesin_diag["eid"].map(pid_to_practice)
    # 丢弃不在 GP 数据库中的患者 (没有 PRACTICE_ID 映射)
    hesin_diag = hesin_diag.dropna(subset=["PRACTICE_ID"])
    hesin_diag["PRACTICE_ID"] = hesin_diag["PRACTICE_ID"].astype(int)

    # 4. 批量插入
    records = list(zip(
        hesin_diag["PRACTICE_ID"],
        hesin_diag["eid"],
        hesin_diag["EVENT"],
        hesin_diag["admidate"]
    ))
    print(f"Inserting {len(records):,} HES diagnosis events...")

    BATCH = 200_000
    for i in tqdm(range(0, len(records), BATCH)):
        cur_hes.executemany(
            "INSERT INTO diagnosis_table VALUES (?, ?, ?, ?)",
            records[i:i+BATCH]
        )
    conn_hes.commit()

    # 5. 建索引
    cur_hes.execute("""
        CREATE INDEX IF NOT EXISTS diagnosis_index
        ON diagnosis_table (PRACTICE_ID)
    """)
    conn_hes.commit()

    # 6. 统计
    cur_hes.execute("SELECT COUNT(*) FROM diagnosis_table")
    n_events = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT PATIENT_ID) FROM diagnosis_table")
    n_patients = cur_hes.fetchone()[0]
    cur_hes.execute("SELECT COUNT(DISTINCT EVENT) FROM diagnosis_table")
    n_codes = cur_hes.fetchone()[0]

    print(f"\nHES Database built:")
    print(f"  Events: {n_events:,}")
    print(f"  Patients: {n_patients:,}")
    print(f"  Unique ICD-10 codes (truncated to {ICD10_TRUNCATE_LEN} chars): {n_codes:,}")
    print(f"  Output: {HES_DB}")

    conn_hes.close()

if __name__ == "__main__":
    main()
```

#### 3.2.3 关键细节

- **PRACTICE_ID 复用**: 从 GP 的 `static_table` 获取每个患者的 `PRACTICE_ID`，这样 HES 数据使用**完全相同的 practice-based train/val/test split**
- **static_table 完全复制**: 从 GP 数据库复制，保证 `_parquet_row_to_static_covariates()` 函数能正常工作
- **ICD-10 截断到 3 位**: 原始 HES 有 12,315 种 ICD-10 码，截断到 3 位后约 1,000-2,000 种，更适合预训练词汇量
- **仅保留 level=1 (主诊断)**: 避免过多副诊断膨胀序列。每次入院约产生 1 个事件

### 3.3 Step 2: 构建 HES Pretrain Dataset

**新建文件**: `CPRD/examples/modelling/SurvivEHR/build_hes_pretrain_dataset.py`

使用 `FoundationalDataModule(load=False)` 从 HES 数据库生成 Parquet 数据集，方式与 GP pretrain 完全相同。

```python
"""
build_hes_pretrain_dataset.py
=============================
Build HES pretrain dataset using FoundationalDataModule.
Uses the HES-only database and GP practice splits for consistent splitting.
"""

import torch
import logging
from FastEHR.dataloader import FoundationalDataModule

HES_DB = "/Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db"
HES_DS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/"
SPLITS = "/Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/practice_id_splits.pickle"

SEED = 1337
NUM_THREADS = 12

def main():
    torch.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    print("Building HES pretrain dataset...")
    dm = FoundationalDataModule(
        path_to_db=HES_DB,
        path_to_ds=HES_DS,
        load=False,
        tokenizer="tabular",
        overwrite_practice_ids=SPLITS,      # 使用与GP相同的practice split
        # 不要 overwrite_meta_information — 让它从HES数据自动生成新的meta info
        min_workers=NUM_THREADS,
        seed=SEED,
    )

    print(f"HES pretrain dataset built:")
    print(f"  Train: {len(dm.train_set)}")
    print(f"  Val:   {len(dm.val_set)}")
    print(f"  Test:  {len(dm.test_set)}")

if __name__ == "__main__":
    main()
```

**重要**: 不要传 `overwrite_meta_information`。让 `FoundationalDataModule` 从 HES 数据自动构建新的 meta_information（新的 tokenizer 词汇表），因为 HES 使用 ICD-10 码，与 GP 的 Read v2 码完全不同。HES 会生成自己的 `meta_information_custom.pickle`。

输出:
- `CPRD/data/FoundationalModel/PreTrain_HES/` — HES pretrain dataset
- `CPRD/data/FoundationalModel/PreTrain_HES/meta_information_custom.pickle` — HES 词汇表和元信息

### 3.4 Step 3: HES Pretrain Config

**新建文件**: `CPRD/examples/modelling/SurvivEHR/confs/config_HES_Pretrain.yaml`

```yaml
is_decoder: True

data:
  batch_size: 64
  unk_freq_threshold: 0.0
  min_workers: 12
  global_diagnoses: False      # HES 序列较短，不需要 global_diagnoses
  repeating_events: True       # HES 中允许重复事件（同一诊断可反复住院）
  path_to_db: /Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db
  path_to_ds: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/
  meta_information_path: null   # 使用自动生成的 meta info
  subsample_training: null
  num_static_covariates: 27     # 与GP相同的static covariates
  supervised_time_scale: 1.0

experiment:
  type: 'pre-train'
  project_name: SurvivEHR
  run_id: ${head.SurvLayer}PreTrain_HES_${experiment.seed}
  fine_tune_id: null
  notes: "HES-only pretrain: ICD-10 next-event prediction"
  tags: ["hes", "pretrain", "icd10"]
  train: True
  test: True
  verbose: True
  seed: 1337
  log: True
  log_dir: /Data0/swangek_data/991/CPRD/output/
  ckpt_dir: /Data0/swangek_data/991/CPRD/output/checkpoints/

fine_tuning:
  fine_tune_outcomes: null
  custom_outcome_method:
    _target_: null
  custom_stratification_method:
    _target_: null
  use_callbacks:
    hidden_embedding:
      num_batches: 0
      mask_static: False
      mask_value: False
    performance_metrics: True
    rmst: False
  compression_layer: False
  llrd: null
  PEFT:
    method: null
    adapter_dim: 8
  backbone:
    linear_probe_epochs: 0
    unfreeze_top_k: null
  head:
    surv_weight: 1
    value_weight: 0
    learning_rate: 5e-4
  sample_weighting:
    mode: null
    event_lambda: 1.0
    alpha: 2.0
    tau: 0.33
    w_t_max: 3.0
    w_total_max: 20.0

optim:
  num_epochs: 15
  learning_rate: 3e-4
  scheduler_warmup: True
  scheduler: decaycawarmrestarts
  scheduler_periods: 5000          # HES数据量较小，warmup步数减少
  learning_rate_decay: 0.8
  val_check_interval: 500          # 验证频率更高（数据量小）
  early_stop: True
  early_stop_patience: 20
  log_every_n_steps: 20
  limit_val_batches: 0.05
  limit_test_batches: null
  accumulate_grad_batches: 2

transformer:
  block_type: "Neo"
  block_size: 256               # HES序列较短，256足够
  n_layer: 6                    # 与GP backbone相同架构
  n_head: 6
  n_embd: 384
  layer_norm_bias: False
  attention_type: "global"
  bias: True
  dropout: 0.0
  attention_dropout: 0.0
  resid_dropout: 0.0
  private_heads: 0

head:
  SurvLayer: "cr"
  surv_weight: 1
  tokens_for_univariate_regression: None
  value_weight: 0.0              # HES 没有数值型测量值
```

**关键差异与 GP pretrain**:
- `block_size: 256`（GP 用 256 pretrain，512 finetune）
- `repeating_events: True`（HES 允许同一诊断反复出现）
- `global_diagnoses: False`（HES 序列较短）
- `value_weight: 0.0`（HES 没有数值型测量值）
- `path_to_db`: 指向 HES 数据库
- `path_to_ds`: 指向 HES pretrain 数据集
- `run_id`: `crPreTrain_HES_1337` → checkpoint 将保存为 `crPreTrain_HES_1337.ckpt`

### 3.5 Step 4: 运行 HES Pretrain

**新建文件**: `CPRD/run_hes_pretrain.sh`

```bash
#!/bin/bash
set -e

PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"

cd "$WORK_DIR"

echo "============================================================"
echo "  HES Backbone Pretrain Pipeline"
echo "  Start: $(date)"
echo "============================================================"

# Step 1: Build HES database
echo "===== Step 1: Build HES database ====="
$PYTHON build_hes_database.py

# Step 2: Build HES pretrain dataset
echo "===== Step 2: Build HES pretrain dataset ====="
$PYTHON build_hes_pretrain_dataset.py

# Step 3: Pretrain HES backbone
echo "===== Step 3: Pretrain HES backbone ====="
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
$PYTHON run_experiment.py --config-name=config_HES_Pretrain

echo "===== HES PRETRAIN DONE ====="
echo "Checkpoint: ${CKPT_DIR}/crPreTrain_HES_1337.ckpt"
```

### 3.6 阶段一预期输出

| 产物 | 路径 |
|------|------|
| HES SQLite 数据库 | `CPRD/data/hes_pretrain_database.db` |
| HES Pretrain 数据集 | `CPRD/data/FoundationalModel/PreTrain_HES/` |
| HES meta info | `CPRD/data/FoundationalModel/PreTrain_HES/meta_information_custom.pickle` |
| HES pretrained checkpoint | `CPRD/output/checkpoints/crPreTrain_HES_1337.ckpt` |

---

## 4. 阶段二：Dual-Backbone Fine-tune

### 4.1 目标

创建 `DualBackboneFineTuneExperiment`，加载两个独立的 pretrained backbone（GP + HES），通过 fusion layer 合并 hidden states，再接 competing-risk survival head。

### 4.2 Step 5: 构建 Dual Fine-tune Dataset

**新建文件**: `CPRD/examples/modelling/SurvivEHR/build_dementia_cr_dual.py`

需要构建一个**同时包含 GP 和 HES 信息**的数据集。由于两个 backbone 各自需要不同格式的输入，有两种实现方式：

**方案 A（推荐）: 两个独立数据集 + 自定义 DataLoader**

为 GP 和 HES 分别构建独立数据集，然后在 DataLoader 级别按 PATIENT_ID 配对。

**方案 B: 扩展 Parquet 增加 HES 列**

在 GP 数据集的 Parquet 中增加 `HES_TOKENS`、`HES_DATES` 列，然后在 `__getitem__` 中分别处理。

**推荐方案 A**，因为两个 backbone 有完全不同的 tokenizer（Read v2 vs ICD-10），独立数据集更干净。

#### 4.2.1 数据集构建逻辑

```python
"""
build_dementia_cr_dual.py
=========================
Build fine-tuning datasets for dual-backbone architecture.
Creates TWO datasets:
  1. GP dataset: 与 hes_static 完全相同 (GP seq + HES labels + HES static features)
     路径: CPRD/data/FoundationalModel/FineTune_Dementia_CR_dual_gp/
  2. HES dataset: 相同患者的 HES 序列（用于 HES backbone 输入）
     路径: CPRD/data/FoundationalModel/FineTune_Dementia_CR_dual_hes/

两个数据集的患者集合完全一致、split 完全一致。
"""

# GP 数据集: 直接复用 build_dementia_cr_hes_static.py 的逻辑
# HES 数据集: 使用 HES database + same practice splits + same study inclusion

# 关键: HES 数据集也需要包含 supervised 信息 (target_token, target_age_delta)
# 但 HES 数据集的 **序列内容** 是 ICD-10 码
# **标签** (dementia/death/censored) 与 GP 数据集完全一致

# 实现:
#   1. 先构建 GP 数据集 (复用 build_dementia_cr_hes_static.py)
#   2. 提取 GP 数据集中的所有 patient_id 及其 split 分配
#   3. 构建 HES 数据集:
#      a) 使用 HES DB + HES pretrain meta_info
#      b) 使用相同的 study inclusion criteria
#      c) 仅保留 GP 数据集中存在的患者
#      d) 应用相同的 label augmentation (HES dementia labels)
```

#### 4.2.2 HES Fine-tune Dataset 的关键细节

HES fine-tune 数据集需要特殊处理，因为 FoundationalDataModule 生成的是 next-event prediction 格式，但我们需要的是 **supervised** 格式（最后一个事件是 outcome）。

**解决方案**: 不要试图用 FoundationalDataModule 直接生成 supervised HES 数据集。而是：

1. 使用 FoundationalDataModule 构建**非 supervised** 的 HES 数据集（存储每个患者的完整 ICD-10 序列）
2. 在 fine-tune 的 DataLoader 中，从 GP 数据集读取 label 信息（target_token, target_age_delta），从 HES 数据集读取 HES 序列

**或者更简单的方案**: 在 Parquet 中直接存储预处理好的 HES token 序列，作为额外的列加入 GP 数据集。

### 4.3 Step 6: 创建 DualBackbone 模型类

**新建文件**: `CPRD/src/models/survival/task_heads/dual_backbone.py`

这是**最核心的修改**。创建一个新的模型类 `DualBackboneSurvModel`，包含两个 `TTETransformer` 和一个 fusion layer。

```python
"""
dual_backbone.py
================
Dual-backbone model: GP Transformer + HES Transformer + Fusion + Survival Head

文件位置: CPRD/src/models/survival/task_heads/dual_backbone.py
"""

import torch
from torch import nn
import logging
from typing import Optional

from SurvivEHR.src.models.TTE.base import TTETransformer


class FusionLayer(nn.Module):
    """将两个backbone的hidden states融合为一个向量"""

    def __init__(self, embed_dim: int, fusion_type: str = "concat_linear"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat_linear":
            # 最简单: concat后线性投影
            self.proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        elif fusion_type == "gated":
            # 门控融合: 学习GP和HES的动态权重
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.proj_gp = nn.Linear(embed_dim, embed_dim)
            self.proj_hes = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, h_gp: torch.Tensor, h_hes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_gp:  (bsz, embed_dim) — GP backbone last hidden state
            h_hes: (bsz, embed_dim) — HES backbone last hidden state
        Returns:
            h_fused: (bsz, embed_dim)
        """
        if self.fusion_type == "concat_linear":
            return self.proj(torch.cat([h_gp, h_hes], dim=-1))
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([h_gp, h_hes], dim=-1))
            return gate * self.proj_gp(h_gp) + (1 - gate) * self.proj_hes(h_hes)


class DualBackboneSurvModel(nn.Module):
    """
    双backbone生存分析模型

    包含:
      - gp_transformer: GP backbone (TTETransformer, 从GP pretrain checkpoint加载)
      - hes_transformer: HES backbone (TTETransformer, 从HES pretrain checkpoint加载)
      - fusion: FusionLayer (从零学习)
    """

    def __init__(
        self,
        cfg,
        gp_vocab_size: int,
        hes_vocab_size: int,
        gp_num_static_covariates: int = 35,   # 27 base + 8 HES features
        hes_num_static_covariates: int = 27,   # HES backbone 只用基础 static
        fusion_type: str = "gated",
    ):
        super().__init__()

        self.n_embd = cfg.transformer.n_embd
        self.block_size = cfg.transformer.block_size

        # GP Backbone
        self.gp_transformer = TTETransformer(
            cfg, gp_vocab_size,
            num_static_covariates=gp_num_static_covariates
        )

        # HES Backbone — 使用可能不同的 block_size
        # 创建一个修改后的 cfg 给 HES backbone
        # 注意: 需要通过 cfg 传入 hes_block_size，或者硬编码
        import copy
        hes_cfg = copy.deepcopy(cfg)
        hes_block_size = getattr(cfg, 'hes_block_size', 256)
        hes_cfg.transformer.block_size = hes_block_size

        self.hes_transformer = TTETransformer(
            hes_cfg, hes_vocab_size,
            num_static_covariates=hes_num_static_covariates
        )

        # Fusion Layer
        self.fusion = FusionLayer(self.n_embd, fusion_type=fusion_type)

    def forward(
        self,
        # GP inputs
        gp_tokens: torch.Tensor,
        gp_ages: torch.Tensor,
        gp_values: torch.Tensor,
        gp_covariates: torch.Tensor,
        gp_attention_mask: torch.Tensor,
        # HES inputs
        hes_tokens: torch.Tensor,
        hes_ages: torch.Tensor,
        hes_values: torch.Tensor,
        hes_covariates: torch.Tensor,
        hes_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            h_fused: (bsz, embed_dim) — fused hidden states 的 last token
        """

        # GP forward
        h_gp_seq = self.gp_transformer(
            tokens=gp_tokens, ages=gp_ages,
            values=gp_values, covariates=gp_covariates,
            attention_mask=gp_attention_mask
        )  # (bsz, seq_len_gp, embed_dim)

        # HES forward
        h_hes_seq = self.hes_transformer(
            tokens=hes_tokens, ages=hes_ages,
            values=hes_values, covariates=hes_covariates,
            attention_mask=hes_attention_mask
        )  # (bsz, seq_len_hes, embed_dim)

        # 注意: h_gp_seq 和 h_hes_seq 的 last token 提取
        # 在 FineTuneExperiment.forward() 中通过 gen_mask 实现

        return h_gp_seq, h_hes_seq
```

### 4.4 Step 7: 创建 DualFineTuneExperiment

**新建文件**: `CPRD/examples/modelling/SurvivEHR/setup_dual_finetune_experiment.py`

这是 `setup_finetune_experiment.py` 的双模型版本。

```python
"""
setup_dual_finetune_experiment.py
=================================
Dual-backbone fine-tuning experiment.

与 setup_finetune_experiment.py 的核心区别:
  1. 模型包含两个 TTETransformer (GP + HES)
  2. forward() 分别处理 GP 和 HES 输入
  3. 从两个 hidden_state 中提取 last token，fusion 后送入 survival head
  4. Checkpoint 加载分两步: GP pretrain ckpt + HES pretrain ckpt

关键类: DualFineTuneExperiment(pl.LightningModule)
"""

import logging
import copy
import pytorch_lightning as pl
import torch
from torch import nn
from omegaconf import OmegaConf

from SurvivEHR.src.models.survival.task_heads.dual_backbone import (
    DualBackboneSurvModel, FusionLayer
)
from SurvivEHR.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from SurvivEHR.examples.modelling.SurvivEHR.setup_finetune_experiment import compute_sample_weights
from SurvivEHR.examples.modelling.SurvivEHR.optimizers_fine_tuning import ConfigureFTOptimizers


class DualFineTuneExperiment(pl.LightningModule):

    def __init__(
        self,
        cfg,
        outcome_tokens,
        risk_model,
        gp_vocab_size: int,
        hes_vocab_size: int,
        outcome_token_groups=None,
        fusion_type: str = "gated",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Sample weighting (复用 FineTuneExperiment 的逻辑)
        sw_cfg = getattr(cfg.fine_tuning, "sample_weighting", None)
        self.weighting_mode = str(getattr(sw_cfg, "mode", "none")) if sw_cfg else "none"
        # ... (复制 FineTuneExperiment.__init__ 中的 sample_weighting 配置代码)

        # === 双 Backbone 模型 ===
        self.model = DualBackboneSurvModel(
            cfg=cfg,
            gp_vocab_size=gp_vocab_size,
            hes_vocab_size=hes_vocab_size,
            gp_num_static_covariates=cfg.data.num_static_covariates,  # 35
            hes_num_static_covariates=27,  # HES backbone 用基础 static
            fusion_type=fusion_type,
        )

        # === Survival Head (与 FineTuneExperiment 相同) ===
        hidden_dim = cfg.transformer.n_embd  # fusion 输出 = embed_dim = 384

        # Competing Risk head
        if outcome_token_groups is not None:
            num_risks = len(outcome_token_groups)
            frozen_groups = [list(g) for g in outcome_token_groups]
            def _grouped_reduce(target_token, _groups=frozen_groups):
                result = torch.zeros_like(target_token)
                for gidx, group in enumerate(_groups):
                    for tid in group:
                        result = torch.where(target_token == tid, gidx + 1, result)
                return result
            self.reduce_to_outcomes = _grouped_reduce
        else:
            num_risks = len(outcome_tokens)
            self.reduce_to_outcomes = lambda target_token: sum(
                [torch.where(target_token == i, idx + 1, 0)
                 for idx, i in enumerate(outcome_tokens)]
            )

        self.surv_layer = ODESurvCompetingRiskLayer(
            hidden_dim, [32, 32], num_risks=num_risks,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, batch, return_loss=True, return_generation=False, is_generation=False):
        """
        batch 需要同时包含 GP 和 HES 的输入:
          - batch['tokens'], batch['ages'], etc. → GP inputs
          - batch['hes_tokens'], batch['hes_ages'], etc. → HES inputs
          - batch['target_token'], batch['target_age_delta'] → labels (来自GP)
        """

        # === GP 输入 ===
        gp_tokens = batch['tokens'].to(self.device)
        gp_ages = batch['ages'].to(self.device)
        gp_values = batch['values'].to(self.device)
        gp_covariates = batch['static_covariates'].to(self.device)
        gp_attention_mask = batch['attention_mask'].to(self.device)

        # === HES 输入 ===
        hes_tokens = batch['hes_tokens'].to(self.device)
        hes_ages = batch['hes_ages'].to(self.device)
        hes_values = batch['hes_values'].to(self.device)
        hes_covariates = batch['hes_static_covariates'].to(self.device)
        hes_attention_mask = batch['hes_attention_mask'].to(self.device)

        # === 目标 ===
        target_token = batch['target_token'].reshape((-1, 1)).to(self.device)
        target_age_delta = batch['target_age_delta'].reshape((-1, 1)).to(self.device)
        bsz = gp_tokens.shape[0]

        # === Forward through both backbones ===
        h_gp_seq, h_hes_seq = self.model(
            gp_tokens=gp_tokens, gp_ages=gp_ages,
            gp_values=gp_values, gp_covariates=gp_covariates,
            gp_attention_mask=gp_attention_mask,
            hes_tokens=hes_tokens, hes_ages=hes_ages,
            hes_values=hes_values, hes_covariates=hes_covariates,
            hes_attention_mask=hes_attention_mask,
        )

        # === 提取 GP last token hidden state ===
        _att_tmp = torch.hstack((gp_attention_mask,
                                  torch.zeros((bsz, 1), device=self.device)))
        gen_mask = gp_attention_mask - _att_tmp[:, 1:]
        h_gp = torch.zeros((bsz, h_gp_seq.shape[-1]), device=self.device)
        for idx in range(bsz):
            h_gp[idx] = h_gp_seq[idx, gen_mask[idx] == 1, :]

        # === 提取 HES last token hidden state ===
        # 对于没有 HES 记录的患者, hes_attention_mask 全0, h_hes 保持零向量
        h_hes = torch.zeros((bsz, h_hes_seq.shape[-1]), device=self.device)
        _att_tmp_hes = torch.hstack((hes_attention_mask,
                                      torch.zeros((bsz, 1), device=self.device)))
        gen_mask_hes = hes_attention_mask - _att_tmp_hes[:, 1:]
        for idx in range(bsz):
            if gen_mask_hes[idx].sum() == 1:
                h_hes[idx] = h_hes_seq[idx, gen_mask_hes[idx] == 1, :]
            # else: h_hes[idx] remains zero — 没有 HES 记录

        # === Fusion ===
        h_fused = self.model.fusion(h_gp, h_hes)  # (bsz, embed_dim)

        # === Survival prediction ===
        # DeSurv 需要 (bsz, seq_len=2, embed_dim) 格式
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
            loss = torch.sum(torch.stack(losses_desurv))
        else:
            loss = None

        outputs = {"surv": surv_dict}
        losses = {"loss": loss, "loss_desurv": loss if loss is not None else None,
                  "loss_values": torch.tensor(0.0)}

        return outputs, losses, h_fused

    def training_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"train_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"val_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def test_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"test_{k}", v, prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']

    def configure_optimizers(self):
        # 使用差异化学习率:
        #   - GP backbone: 最低 LR (已预训练)
        #   - HES backbone: 最低 LR (已预训练)
        #   - Fusion layer + Survival head: 较高 LR (从零学习)
        params = [
            {"params": self.model.gp_transformer.parameters(),
             "lr": self.cfg.optim.learning_rate},        # 5e-5
            {"params": self.model.hes_transformer.parameters(),
             "lr": self.cfg.optim.learning_rate},        # 5e-5
            {"params": self.model.fusion.parameters(),
             "lr": self.cfg.fine_tuning.head.learning_rate},  # 5e-4
            {"params": self.surv_layer.parameters(),
             "lr": self.cfg.fine_tuning.head.learning_rate},  # 5e-4
        ]
        optimizer = torch.optim.AdamW(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
```

### 4.5 Step 8: 自定义 DataLoader

**核心挑战**: batch 中需要同时包含 GP 输入和 HES 输入。

**方案**: 创建一个 `DualDataModule`，在 `collate_fn` 中同时加载 GP 和 HES 数据。

**新建文件**: `CPRD/examples/modelling/SurvivEHR/dual_data_module.py`

```python
"""
dual_data_module.py
===================
DataModule that yields batches containing both GP and HES inputs for each patient.

实现思路:
  - 使用 GP FoundationalDataModule 作为主 DataModule (提供 patient 遍历和 labels)
  - 维护一个 HES 序列的内存缓存 (patient_id -> hes_tokens, hes_ages)
  - 在 collate_fn 中, 为每个 GP batch item 附加对应的 HES 序列

关键修改点:
  foundational_loader.py 中的 FoundationalDataset.__getitem__() 返回:
    {
      "static_covariates": ...,
      "tokens": ...,
      "ages": ...,
      "values": ...,
      # supervised 模式额外包含:
      "target_token": ...,
      "target_age_delta": ...,
      "target_value": ...,
    }

  我们需要在 collate_fn 中添加:
    {
      "hes_tokens": ...,
      "hes_ages": ...,
      "hes_values": ...,        # 全 NaN (HES 没有数值)
      "hes_static_covariates": ...,   # 27-dim (基础 static)
      "hes_attention_mask": ...,
    }
"""

import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class DualCollateWrapper:
    """
    包装现有的 collate_fn, 为每个 batch 添加 HES 输入。

    Usage:
        original_collate = dm.collate_fn
        hes_cache = load_hes_cache(...)    # {patient_id: {"tokens": [...], "ages": [...]}}
        dm.collate_fn = DualCollateWrapper(original_collate, hes_cache, hes_tokenizer, hes_block_size=256)
    """

    def __init__(self, original_collate, hes_cache, hes_tokenizer, hes_block_size=256):
        self.original_collate = original_collate
        self.hes_cache = hes_cache              # {pid: {"tokens": list[str], "ages": list[float]}}
        self.hes_tokenizer = hes_tokenizer
        self.hes_block_size = hes_block_size
        # 复用 original_collate 的属性以维持兼容性
        self.supervised = original_collate.supervised
        self.supervised_time_scale = getattr(original_collate, 'supervised_time_scale', 1.0)

    def __call__(self, batch_items):
        # 1. 先用原始 collate 处理 GP 数据
        gp_batch = self.original_collate(batch_items)

        # 2. 为每个 patient 构建 HES 输入
        bsz = gp_batch['tokens'].shape[0]
        hes_tokens_list = []
        hes_ages_list = []

        for item in batch_items:
            pid = int(item.get("PATIENT_ID", -1))
            hes_data = self.hes_cache.get(pid, None)

            if hes_data is not None and len(hes_data["tokens"]) > 0:
                # 编码 HES 序列
                raw_tokens = hes_data["tokens"]
                raw_ages = hes_data["ages"]

                # 截断到 hes_block_size
                if len(raw_tokens) > self.hes_block_size:
                    raw_tokens = raw_tokens[-self.hes_block_size:]
                    raw_ages = raw_ages[-self.hes_block_size:]

                encoded = self.hes_tokenizer.encode(raw_tokens)
                hes_tokens_list.append(torch.tensor(encoded))
                hes_ages_list.append(torch.tensor(raw_ages, dtype=torch.float))
            else:
                # 没有 HES 记录 — 空序列 (会被 pad 为全0, attention_mask 全0)
                hes_tokens_list.append(torch.tensor([], dtype=torch.long))
                hes_ages_list.append(torch.tensor([], dtype=torch.float))

        # 3. Pad HES 序列
        if all(len(t) == 0 for t in hes_tokens_list):
            # 极端情况: 整个 batch 都没有 HES 记录
            hes_tokens_padded = torch.zeros((bsz, 1), dtype=torch.long)
            hes_ages_padded = torch.zeros((bsz, 1), dtype=torch.float)
            hes_attention_mask = torch.zeros((bsz, 1), dtype=torch.float)
        else:
            hes_tokens_padded = pad_sequence(
                [t if len(t) > 0 else torch.tensor([0]) for t in hes_tokens_list],
                batch_first=True, padding_value=0
            )
            hes_ages_padded = pad_sequence(
                [a if len(a) > 0 else torch.tensor([0.0]) for a in hes_ages_list],
                batch_first=True, padding_value=0.0
            )
            hes_attention_mask = (hes_tokens_padded != 0).float()
            # 修正: 真正的空序列应全部mask
            for i, t in enumerate(hes_tokens_list):
                if len(t) == 0:
                    hes_attention_mask[i] = 0

        hes_values_padded = torch.full_like(hes_ages_padded, float('nan'))

        # HES static covariates: 使用 GP 的前 27 维
        hes_static = gp_batch['static_covariates'][:, :27].clone()

        # 4. 合并到 batch
        gp_batch['hes_tokens'] = hes_tokens_padded
        gp_batch['hes_ages'] = hes_ages_padded
        gp_batch['hes_values'] = hes_values_padded
        gp_batch['hes_static_covariates'] = hes_static
        gp_batch['hes_attention_mask'] = hes_attention_mask

        return gp_batch
```

### 4.6 Step 9: 修改 run_experiment.py

需要在 `run_experiment.py` 中添加 `"dualfinetune"` case，或者创建一个新的入口文件。

**推荐**: 创建新文件 `run_dual_experiment.py`，避免对现有代码造成风险。

**新建文件**: `CPRD/examples/modelling/SurvivEHR/run_dual_experiment.py`

```python
"""
run_dual_experiment.py
======================
Entry point for dual-backbone fine-tuning.

与 run_experiment.py 的区别:
  1. 加载两个 DataModule (GP + HES) 或一个 DualDataModule
  2. 从两个 pretrain checkpoint 加载 backbone weights
  3. 使用 DualFineTuneExperiment 而不是 FineTuneExperiment
"""

# 核心流程:
# 1. 加载 GP DataModule (supervised, 使用 hes_static 数据集)
# 2. 加载 HES pretrain meta_info → 构建 HES tokenizer
# 3. 构建 HES 序列缓存 (从 HES DB 读取每个患者的 ICD-10 序列)
# 4. 用 DualCollateWrapper 包装 GP DataModule 的 collate_fn
# 5. 创建 DualFineTuneExperiment
# 6. 加载 GP pretrain weights → gp_transformer
# 7. 加载 HES pretrain weights → hes_transformer
# 8. Train + Eval
```

### 4.7 Step 10: Checkpoint 加载逻辑

**关键代码**: 分别从两个 pretrain checkpoint 加载 backbone weights

```python
def load_dual_pretrained_weights(
    model: DualBackboneSurvModel,
    gp_ckpt_path: str,
    hes_ckpt_path: str,
):
    """
    从两个独立的 pretrain checkpoint 加载 backbone weights。

    GP checkpoint 结构 (CausalExperiment):
      state_dict keys: "model.transformer.wte.*", "model.transformer.wpe.*",
                       "model.transformer.blocks.*", "model.transformer.ln_f.*",
                       "model.surv_layer.*", "model.value_layer.*"
      我们只需要 "model.transformer.*" 部分

    加载到 DualBackboneSurvModel:
      "model.transformer.*" → "gp_transformer.*"

    HES checkpoint 同理:
      "model.transformer.*" → "hes_transformer.*"
    """

    # --- GP backbone ---
    gp_ckpt = torch.load(gp_ckpt_path, map_location='cpu')
    gp_sd = gp_ckpt.get('state_dict', gp_ckpt)

    gp_mapping = {}
    for k, v in gp_sd.items():
        # CausalExperiment 中: "model.transformer.xxx"
        # 目标: "gp_transformer.xxx"
        if k.startswith("model.transformer."):
            new_key = k.replace("model.transformer.", "gp_transformer.")
            gp_mapping[new_key] = v

    # 处理 static_proj size mismatch (27 → 35)
    model_sd = model.state_dict()
    for k, v in list(gp_mapping.items()):
        if k in model_sd and v.shape != model_sd[k].shape:
            if 'static_proj' in k:
                new_p = model_sd[k].clone()
                if v.dim() == 2:
                    new_p[:v.shape[0], :v.shape[1]] = v
                else:
                    new_p[:v.shape[0]] = v
                gp_mapping[k] = new_p
                logging.info(f"GP partial load {k}: {v.shape} → {model_sd[k].shape}")
            else:
                logging.warning(f"GP skip {k}: {v.shape} vs {model_sd[k].shape}")
                del gp_mapping[k]

    # --- HES backbone ---
    hes_ckpt = torch.load(hes_ckpt_path, map_location='cpu')
    hes_sd = hes_ckpt.get('state_dict', hes_ckpt)

    hes_mapping = {}
    for k, v in hes_sd.items():
        if k.startswith("model.transformer."):
            new_key = k.replace("model.transformer.", "hes_transformer.")
            hes_mapping[new_key] = v

    # HES backbone 的 static_proj 应该是 (384, 27) → (384, 27)，不需要 resize
    for k, v in list(hes_mapping.items()):
        if k in model_sd and v.shape != model_sd[k].shape:
            logging.warning(f"HES skip {k}: {v.shape} vs {model_sd[k].shape}")
            del hes_mapping[k]

    # --- 合并加载 ---
    combined = {**gp_mapping, **hes_mapping}
    missing, unexpected = model.load_state_dict(combined, strict=False)

    logging.info(f"Loaded dual pretrained weights:")
    logging.info(f"  GP backbone: {len(gp_mapping)} keys")
    logging.info(f"  HES backbone: {len(hes_mapping)} keys")
    logging.info(f"  Missing (new layers): {len(missing)} keys")
    logging.info(f"  Unexpected: {len(unexpected)} keys")

    # Missing keys 应该只有 fusion layer 和 surv_layer
    return model
```

### 4.8 Step 11: Fine-tune Config

**新建文件**: `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual.yaml`

```yaml
is_decoder: True

data:
  batch_size: 16                    # 双 backbone 显存更大，减小 batch
  unk_freq_threshold: 0.0
  min_workers: 12
  global_diagnoses: True
  repeating_events: False
  path_to_db: /Data0/swangek_data/991/CPRD/data/example_exercise_database.db
  path_to_ds: /Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_hes_static/
  meta_information_path: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain/meta_information_custom.pickle
  subsample_training: null
  num_static_covariates: 35
  supervised_time_scale: 5.0

# HES-specific data config (新增字段)
hes_data:
  path_to_db: /Data0/swangek_data/991/CPRD/data/hes_pretrain_database.db
  meta_information_path: /Data0/swangek_data/991/CPRD/data/FoundationalModel/PreTrain_HES/meta_information_custom.pickle
  hes_block_size: 256

experiment:
  type: 'dual-fine-tune'           # 新的 experiment type
  project_name: SurvivEHR
  run_id: crPreTrain_small_1337
  fine_tune_id: FineTune_Dementia_CR_dual
  notes: "Dual backbone: GP + HES, gated fusion, idx=72, no SAW"
  tags: ["dementia", "fine-tune", "competing-risk", "dual-backbone", "gated-fusion"]
  train: True
  test: True
  verbose: True
  seed: 1337
  log: True
  log_dir: /Data0/swangek_data/991/CPRD/output/
  ckpt_dir: /Data0/swangek_data/991/CPRD/output/checkpoints/

# 双模型特有配置
dual:
  gp_ckpt: /Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_small_1337.ckpt
  hes_ckpt: /Data0/swangek_data/991/CPRD/output/checkpoints/crPreTrain_HES_1337.ckpt
  fusion_type: "gated"             # "concat_linear" 或 "gated"

fine_tuning:
  fine_tune_outcomes:
    - - "F110."
      - "Eu00."
      - "Eu01."
      # ... (与 hes_static 完全相同的 31 个 dementia codes)
      - "E0020"
    - - "DEATH"

  custom_outcome_method:
    _target_: null
  custom_stratification_method:
    _target_: null

  use_callbacks:
    hidden_embedding:
      num_batches: 0
      mask_static: False
      mask_value: False
    performance_metrics: True
    rmst: False

  compression_layer: False
  llrd: null

  PEFT:
    method: null
    adapter_dim: 8
  backbone:
    linear_probe_epochs: 0
    unfreeze_top_k: null
  head:
    surv_weight: 1
    value_weight: 0
    learning_rate: 5e-4

  sample_weighting:
    mode: null
    event_lambda: 1.0
    alpha: 2.0
    tau: 0.33
    w_t_max: 3.0
    w_total_max: 20.0

optim:
  num_epochs: 25
  learning_rate: 5e-5
  scheduler_warmup: False
  scheduler: reduceonplateau
  scheduler_periods: 10000
  learning_rate_decay: 0.8
  val_check_interval: 1.0
  early_stop: True
  early_stop_patience: 10
  log_every_n_steps: 20
  limit_val_batches: 1.0
  limit_test_batches: null
  accumulate_grad_batches: 32       # 更大梯度累积 (batch_size=16, 需要更多累积)

transformer:
  block_type: "Neo"
  block_size: 512
  n_layer: 6
  n_head: 6
  n_embd: 384
  layer_norm_bias: False
  attention_type: "global"
  bias: True
  dropout: 0.1
  attention_dropout: 0.1
  resid_dropout: 0.1
  private_heads: 0

head:
  SurvLayer: "cr"
  surv_weight: 1
  tokens_for_univariate_regression: None
  value_weight: 0.1
```

### 4.9 Step 12: Pipeline 脚本

**新建文件**: `CPRD/run_dual_pipeline.sh`

```bash
#!/bin/bash
set -e

PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"

cd "$WORK_DIR"

echo "============================================================"
echo "  Dual-Backbone Fine-tuning Pipeline"
echo "  GP backbone: crPreTrain_small_1337.ckpt"
echo "  HES backbone: crPreTrain_HES_1337.ckpt"
echo "  Start: $(date)"
echo "============================================================"

# Step 1: Train
export CUDA_VISIBLE_DEVICES=0
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt" 2>/dev/null || true
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
$PYTHON run_dual_experiment.py --config-name=config_FineTune_Dementia_CR_dual \
    optim.accumulate_grad_batches=32

# Step 2: Eval
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py \
    --config-name=config_FineTune_Dementia_CR_dual_eval

echo "===== DUAL BACKBONE DONE ====="
```

---

## 5. 具体修改文件清单

### 5.1 阶段一（HES Pretrain）— 新建文件

| # | 文件 | 用途 | 复杂度 |
|---|------|------|--------|
| 1 | `CPRD/examples/modelling/SurvivEHR/build_hes_database.py` | 从 CSV 构建 HES SQLite DB | 低 |
| 2 | `CPRD/examples/modelling/SurvivEHR/build_hes_pretrain_dataset.py` | 生成 HES Pretrain Parquet | 低 |
| 3 | `CPRD/examples/modelling/SurvivEHR/confs/config_HES_Pretrain.yaml` | HES 预训练配置 | 低 |
| 4 | `CPRD/run_hes_pretrain.sh` | HES 预训练 pipeline | 低 |

### 5.2 阶段二（Dual Fine-tune）— 新建文件

| # | 文件 | 用途 | 复杂度 |
|---|------|------|--------|
| 5 | `CPRD/src/models/survival/task_heads/dual_backbone.py` | DualBackboneSurvModel + FusionLayer | **高** |
| 6 | `CPRD/examples/modelling/SurvivEHR/setup_dual_finetune_experiment.py` | DualFineTuneExperiment | **高** |
| 7 | `CPRD/examples/modelling/SurvivEHR/dual_data_module.py` | DualCollateWrapper | **中** |
| 8 | `CPRD/examples/modelling/SurvivEHR/run_dual_experiment.py` | 双模型训练入口 | **中** |
| 9 | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual.yaml` | 训练配置 | 低 |
| 10 | `CPRD/examples/modelling/SurvivEHR/confs/config_FineTune_Dementia_CR_dual_eval.yaml` | 评估配置 | 低 |
| 11 | `CPRD/run_dual_pipeline.sh` | 完整 pipeline | 低 |

### 5.3 需要修改的现有文件

| 文件 | 修改内容 | 风险 |
|------|---------|------|
| **无** | 所有新功能通过新文件实现 | 零风险 — 不影响现有实验 |

**设计原则**: 所有改动都是**新文件**，不修改任何现有文件。这保证了 hes_static 和其他已有实验完全不受影响。

---

## 6. 验证方案

### 6.1 阶段一验证

| 检查项 | 预期 |
|--------|------|
| HES DB `diagnosis_table` 行数 | ~4.2M (仅 level=1 主诊断) |
| HES DB 独立 EVENT 数 | ~1,000-2,000 (ICD-10 3位截断) |
| HES Pretrain Dataset 患者数 | ≤449,095 (有 PRACTICE_ID 映射的) |
| HES Pretrain val_loss 下降 | 应正常收敛 |
| `crPreTrain_HES_1337.ckpt` 存在 | 是 |

### 6.2 阶段二验证

| 检查项 | 预期 |
|--------|------|
| 日志显示两个 backbone 权重加载成功 | GP: ~N keys, HES: ~M keys |
| 日志显示 fusion layer missing keys | 是 (从零学习) |
| Batch 中 `hes_tokens` shape | (bsz, ≤256) |
| 无 HES 患者的 `hes_attention_mask` 全 0 | 是 |
| val_loss 正常下降 | 是 |
| **Dementia C_td** | 目标: ≥0.836 (超过 hes_static) |

### 6.3 消融实验

建成后可做以下消融确认各组件贡献：

| 实验 | 设置 | 目的 |
|------|------|------|
| Dual (gated) | 完整双模型 + gated fusion | 主实验 |
| Dual (concat) | concat_linear fusion | 比较 fusion 方式 |
| GP-only (control) | 冻结 HES backbone，只用 GP | 确认 HES backbone 有贡献 |
| HES-only | 冻结 GP backbone，只用 HES | 确认 GP backbone 有贡献 |

---

## 7. 风险与备选方案

### 7.1 显存风险

两个 backbone 的参数量约是单 backbone 的 2 倍。

- GP backbone: ~10M 参数
- HES backbone: ~10M 参数 (vocab 更小，但架构相同)
- Fusion + Head: ~0.5M 参数

**缓解**:
- `batch_size: 16` + `accumulate_grad_batches: 32` (有效 batch = 512)
- HES `block_size: 256` (比 GP 的 512 小)
- 必要时可冻结部分 backbone layers

### 7.2 HES 数据稀疏风险

部分患者可能没有 HES 记录（不在 HES 数据集中）。

**缓解**: h_hes 设为零向量，fusion layer 的 gated 机制可以学到忽略零向量。训练时建议监控有 HES 和无 HES 患者的 loss 分布。

### 7.3 HES 序列太短

如果只保留 level=1 主诊断，每次入院只有 1 个事件，患者住院 5 次 = 5 个 token，序列很短。

**缓解方案**:
- 扩展到 level≤2 (主+副诊断)，每次入院可能有 5-10 个事件
- 加入入院事件本身（如 `ADMISSION` token）作为分隔符
- 加入操作码（从 `hesin_oper.csv`）

### 7.4 备选方案（如果双模型效果不好）

1. **Cross-attention**: 不做 late fusion，让 GP 序列的 token attend to HES 序列的所有 token
2. **HES embedding**: 用预训练 HES backbone 生成固定 embedding（不微调 HES backbone），作为额外 static features
3. **更多 HES static features**: 在现有 8 features 基础上扩展到 50+，覆盖更多 ICD-10 章节

---

## 附录 A: 关键代码路径参考

### 数据流 (Fine-tune forward pass)

```
setup_finetune_experiment.py:216  FineTuneExperiment.forward()
  → batch["tokens"] etc.
  → setup_finetune_experiment.py:233  self.model.transformer(tokens, ages, values, covariates, attention_mask)
    → base.py:88                      TTETransformer.forward()
      → data_embedding_layer.py:74    DataEmbeddingLayer.forward(tokens, values, covariates)
        → data_embedding_layer.py:77  static_proj(covariates).unsqueeze(1) + dynamic_embedding(tokens, values)
      → base.py:91                    wpe(tokens, ages)
      → base.py:94                    x = tok_emb + pos_emb
      → base.py:99-100               for block in self.blocks: x = block(x)
      → base.py:102                   x = ln_f(x)
  → setup_finetune_experiment.py:243-251  提取 last token hidden state
  → setup_finetune_experiment.py:295  surv_layer.predict(in_hidden_state, ...)
```

### Checkpoint key 映射

```
CausalExperiment state_dict:
  "model.transformer.wte.static_proj.weight"     → shape (384, 27)
  "model.transformer.wte.static_proj.bias"        → shape (384,)
  "model.transformer.wte.dynamic_embedding_layer.*"
  "model.transformer.wpe.*"
  "model.transformer.blocks.0-5.*"
  "model.transformer.ln_f.*"
  "model.surv_layer.*"              ← 不需要
  "model.value_layer.*"             ← 不需要

DualBackboneSurvModel state_dict:
  "gp_transformer.wte.static_proj.weight"         → shape (384, 35) ← 需要 partial load
  "gp_transformer.wte.static_proj.bias"           → shape (384,)
  "gp_transformer.wte.dynamic_embedding_layer.*"
  "gp_transformer.wpe.*"
  "gp_transformer.blocks.0-5.*"
  "gp_transformer.ln_f.*"
  "hes_transformer.wte.static_proj.weight"        → shape (384, 27)
  "hes_transformer.wte.dynamic_embedding_layer.*"  ← 不同 vocab, 需要大小匹配
  "hes_transformer.wpe.*"
  "hes_transformer.blocks.0-5.*"
  "hes_transformer.ln_f.*"
  "fusion.proj.*" / "fusion.gate.*"               ← 新层, 从零学习
```

**注意**: `hes_transformer.wte.dynamic_embedding_layer` 的 `nn.Embedding` 大小由 HES vocab size 决定（~2000），与 GP 的 108,118 不同。两个 checkpoint 的 embedding 权重维度不同，这是正常的——各自从各自的 pretrain checkpoint 加载。

### 关键文件行号参考

| 文件 | 行号 | 内容 |
|------|------|------|
| `run_experiment.py:64` | `cfg.data.num_static_covariates = ...shape[1]` | 自动读取 static 维度 |
| `run_experiment.py:117-154` | `case "pretrain"` | 预训练入口 |
| `run_experiment.py:229-281` | `case "finetune"` | 微调入口 |
| `run_experiment.py:289-293` | `last.ckpt` 自动恢复 | **务必删除 last.ckpt** |
| `setup_finetune_experiment.py:131-132` | 模型创建 | `SurvStreamGPTForCausalModelling(cfg, vocab_size, ...)` |
| `setup_finetune_experiment.py:529-560` | `load_from_pretrain` | checkpoint 加载 + static_proj resize |
| `setup_causal_experiment.py:57-70` | `CausalExperiment.__init__` | 预训练模型创建 |
| `setup_causal_experiment.py:262-269` | checkpoint 加载 | `CausalExperiment.load_from_checkpoint(...)` |
| `foundational_loader.py:598-606` | repeating_events 去重 | 只保留最后一次出现 |
| `foundational_loader.py:622-653` | 截断 + global_diagnoses | `start_pos = len - max_seq_length` |
| `foundational_loader.py:662-738` | `_parquet_row_to_static_covariates` | 27-dim + HES_* 列 |
| `data_embedding_layer.py:29` | `static_proj` | `nn.Linear(num_static_covariates, embed_dim)` |
| `data_embedding_layer.py:77` | static embedding 加到 token embedding | `embedded += static.unsqueeze(1)` |
| `causal.py:37` | `TTETransformer` 创建 | `num_static_covariates` 传入 |
| `base.py:33` | `DataEmbeddingLayer` 创建 | `num_static_covariates` 传入 |

---

## 附录 B: 实施优先级

```
Week 1: 阶段一
  Day 1-2: build_hes_database.py + build_hes_pretrain_dataset.py
  Day 2-3: config_HES_Pretrain.yaml + 测试 HES pretrain 运行
  Day 3-7: 运行 HES pretrain (可能需要多天)

Week 2: 阶段二
  Day 1-2: dual_backbone.py (DualBackboneSurvModel + FusionLayer)
  Day 2-3: setup_dual_finetune_experiment.py (DualFineTuneExperiment)
  Day 3-4: dual_data_module.py (DualCollateWrapper)
  Day 4-5: run_dual_experiment.py + config + pipeline 脚本
  Day 5-7: 运行 dual fine-tune + 评估
```
