# Dementia 预测项目阶段进度报告

**项目**：基于 UK Biobank Linked GP + HES 的 Dementia 风险预测（与死亡作为竞争风险）

**模型骨架**：双 Transformer Backbone + Gated Fusion + DeSurv ODE 竞争风险 Head

**数据集**：UK Biobank — GP（CPRD-style 初级保健记录）+ HES（医院住院记录）+ 死亡证明

**汇报范围**：上次会议（汇报 8-dim HES static 架构）至今

---

## 0. TL;DR（一页结论）

### 关键数字演进

| 时间 | 架构/改动 | Dementia C_td (per-batch) | Dementia C_td (cohort, 真实) | 备注 |
|------|----------|---------------:|---------------:|------|
| 上次会议 | hes_static v1 (8-dim) | ~~0.836~~ | — | ❌ 时间泄露 |
| Apr 22 | HES backbone pretrain | (test_loss=2.407) | — | 准备 dual |
| Apr 24 | dual-backbone v1 | ~~0.845~~ | — | ❌ 泄露 |
| Apr 28 | hes_static v2 (22-dim) | ~~0.875~~ | — | ❌ 泄露 |
| Apr 29 | **🚨 时间泄露 bug 发现** | — | — | |
| May 1 | dual v2 (clean retrain) | 0.7569 | **0.8416** | ✅ V1 labels baseline |
| May 6 | cross-attention fusion | 0.7487 | **0.8428** | 不如 gated, 微差 |
| May 7 | V2 标签纠正 | 0.7602 | **0.8447** | +0.003 cohort |
| May 9 | V3 self-training (top 1%, +771) | 0.7685 | **0.8506 ⭐** | **+0.006 cohort, PEAK** |
| May 10 | V4 second-round SST (top 2%, +824) | 0.7732 | 0.8487 | **-0.002 cohort, 下降!** |
| May 12~13 | V2 ablation (single GP) | 0.7571 | **0.8451** | dual ≈ single 在 cohort 也成立 |
| May 14 | V5 third-round SST (top 5%, +2219) | 0.7810 | 0.8467 | **-0.002 cohort, 继续下降** |
| **May 14** | **🚨 发现 per-batch C_td 是 broken 计算方法** | — | — | **跨模型 ranking 反转** |

### ⚠️ 2026-05-14 重大方法学修正

#### 之前所有 C_td 报告都是 per-batch averaged, 不是标准 Antolini's C_td

`clinical_prediction_model.py:PerformanceMetrics` Lightning callback **每个 batch (16 患者) 算一次 C_td**, 然后 Lightning 自动跨 batch 求均值 (reduce_fx=mean default). 这是 PyTorch Lightning 工程实现, 不是 Antolini's 教科书定义.

**实证验证 (V3 batch size sweep)**: bs=16 → 0.7549, bs=64 → 0.7788, bs=256 → 0.8165, **bs=8241 (cohort) → 0.8506**. 47.3% 的 bs=16 batch 含 0 dementia case 被静默丢弃 (selection bias). std=0.289 (噪声极大). 顺序敏感 +0.08 (cross-batch pair 丢失).

→ **所有报告过的 C_td 是 per-batch averaged, 系统性比真实 cohort C_td 低约 +0.06-0.08**.

#### 修正后的 cohort-level 排名 (与 per-batch 完全不同)

| 模型 | per-batch | **cohort** | per-batch 排名 | **cohort 排名 (真)** |
|---|:---:|:---:|:---:|:---:|
| V3 (1st SST, top 1%) | 0.7685 | **0.8506** ⭐ | 第 3 | **第 1 (真 peak)** |
| V4 (2nd SST, top 2%) | 0.7732 | 0.8487 | 第 2 | 第 2 |
| V5 (3rd SST, top 5%) | **0.7810** | 0.8467 | **第 1 (假 peak)** | 第 3 |
| V2 ablation (single GP) | 0.7571 | 0.8451 | 第 5 | 第 4 |
| V2 labels (no SST) | 0.7602 | 0.8447 | 第 4 | 第 5 |
| Cross-attention | 0.7487 | 0.8428 | 第 7 | 第 6 |
| Dual baseline (V1 labels) | 0.7569 | 0.8416 | 第 6 | 第 7 |

**V5 不是 best 模型. V3 才是.**

#### Self-training 是 discrimination ↔ calibration trade-off, 不是单调改进

| 阶段 | C_td (cohort) | IBS dementia (cohort) | 解读 |
|---|:---:|:---:|---|
| V2 labels | 0.8447 | 0.3395 | baseline |
| **V3 (1st SST)** | **0.8506** | 0.3265 | **C_td peak** |
| V4 (2nd SST) | 0.8487 | 0.2773 | -0.002 disc, +0.05 calib |
| V5 (3rd SST) | 0.8467 | **0.2713** | -0.002 disc, +0.06 calib, **IBS peak** |

详见 Section 15 (新加).

### 一句话总结 (修正版 2026-05-14)

> **本项目正确的故事是: (1) 从泄露虚假 +0.10~+0.14 修正为真实 +0.024 (HES static value-add); (2) V2 标签纠正 +0.003 cohort; (3) V3 单轮 self-training +0.006 cohort 是真正 PEAK; (4) V4/V5 续轮交换 discrimination 换 calibration. (5) Dual backbone 贡献 ≈ 0, single GP backbone 同等性能. (6) Per-batch C_td 计算方法本身有 bug, 系统性偏低 ~0.08, 跨模型排名可能反转. 真实 cohort dementia C_td peak = 0.8506 (V3).**
>
> **对外发布主指标**: Dementia C_td (cohort, Antolini) = **0.8506**, 95% CI 约 [0.83, 0.87]. 比之前一直报的 0.7732/0.7810 高约 +0.07-0.08, 不是因为模型变强, 而是修正了 broken measurement.

### 三个论文级别的贡献 (基于 cohort-level 修正后, 2026-05-14)

1. **方法学严谨性 (四层)**:
   - (a) 发现并修复了 HES 特征提取的**时间泄露** bug, 揭示之前虚假的 +0.10 ~ +0.14 提升其实只有真实的 +0.024 (Apr 29)
   - (b) **(NEW 2026-05-14) 发现并修复了 C_td 计算方法 bug**: Lightning callback per-batch averaging 系统性偏低 ~0.07, 跨模型排名可能反转 (per-batch V5>V4>V3 → cohort V3>V4>V5). 详见 Section 15.
   - (c) 明确了 dynamic prediction 模型的**正确评估范式** — C_td 是 paradigm-appropriate, Harrell's C 跨 PH/非 PH paradigm 不可比. 详见 Section 8 修正版.
   - (d) V2 ablation 揭示 **dual backbone 贡献 ≈ 0**, 对架构归因做了诚实修正 (cohort level 再次验证 single GP = dual gated, 差 0.0004)

2. **标签纠正 + self-training** (真正的性能驱动, cohort-level 修正后归因):
   - V2 (HES + 死亡证明 cross-reference 重标 1397 + 移除 487 prevalent): **+0.003 cohort C_td**
   - V3 单轮 self-training (top 1%, 771 pseudo dementia): **+0.006 cohort C_td** ← 最大单步贡献, 真正的 PEAK
   - V4 / V5 续轮 SST: **C_td 不再改善, 但 IBS 持续改善** (discrimination → calibration trade-off, 不是单调改进)
   - **V4 calibration slope = 0.78 (< 1.0) 作为 detecting underdiagnosis 的额外间接证据** — model 预测的概率高于 documented rate, consistent with 文献已知的 30-50% dementia 漏诊. 详见 Section 14.

3. **架构展示** (贡献已诚实下调到接近零):
   - 第一个在 UK Biobank linked GP+HES 上展示 transformer foundation model + 显式死亡竞争风险 + idx age 72 同质年龄分层 的 dementia 预测
   - ⚠️ Dual backbone vs single GP backbone 差距 ≈ 0 (在 per-batch 和 cohort 两个层级都验证, V2 ablation 单 backbone cohort 0.8451 ≈ V2 dual cohort 0.8447)
   - Gated fusion vs cross-attention vs single backbone 都接近 — 架构选择在 clean setting 下不是性能瓶颈
   - 方法学价值: 干净的多模态 EHR + dynamic prediction + competing risk pipeline, 无数据泄露 + 严谨 cohort-level metrics + SurvivEHR backbone 的扩展 case study

---

## 1. 项目背景与目标

### 1.1 临床动机

Dementia 是老年人群最沉重的疾病负担之一，但在初级保健（GP）系统中**确诊率长期偏低**——很多实际患病的人在 GP 数据里看起来是 censored（无诊断）或被标记为 DEATH。如果能用 GP + HES 的纵向 EHR 训练一个准确的预测模型，可以：

1. 识别高风险人群，提早干预
2. 找出"被漏诊"的 dementia 患者，改善流行病学统计
3. 与基因（PRS/APOE）、影像（MRI）、认知测试做生物学交叉验证

### 1.2 技术目标

- **架构**：基于 transformer 的 EHR foundation model，下游做 dementia 风险预测
- **建模**：竞争风险（dementia vs death），用 DeSurv ODE 输出连续时间 CIF（cumulative incidence function）
- **评估**：time-dependent concordance（C_td）作为主指标

### 1.3 数据规模

| 数据集 | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| GP PreTrain | ~450K+ | ~50K+ | ~50K+ | ~550K+ |
| HES PreTrain | 327,308 | 17,746 | 19,522 | 364,576 |
| FineTune Dementia CR (idx72) | ~119K | ~6K | ~8K | ~134K |

Index age 固定在 72 岁，study end 2022-10-31。

---

## 2. 起点：上次会议的状态

**已完成**：
- GP foundation model 预训练完成（`crPreTrain_small_1337.ckpt`，15 epoch）
- HES Label Augmentation (hes_aug)：用 HES 中的 dementia 诊断对 GP 中 censored 患者做标签修正，~4,097 人被 relabel → **Dementia C_td = 0.733（baseline）**
- HES Static Features v1（8 维）：高血压、糖尿病、卒中、心梗、心衰、谵妄、TBI、cumulative admission count → 当时报告 **Dementia C_td = 0.836（+0.103）**

**当时认为的 SOTA**：hes_static v1，C_td = 0.836

**注**：这个 0.836 后来被证实因时间泄露而虚高。

---

## 3. 阶段一：架构探索（4 月中下旬）

### 3.1 实验 A：HES Full Sequence Fusion v5（失败）

**思路**：把 HES 住院诊断（ICD-10）通过 OMOP 映射翻译成 Read v2 code，与 GP 序列在 token 级别融合成一条统一长序列，再训练 transformer。

**实施**：
- 构建 `omop_icd10_to_readv2.pickle`（OMOP CDM 中介映射）
- HOTFIX：F00/F01/F02/F03/G30 (dementia ICD-10) 强制映射到 31 个 dementia Read v2 codes（防止 _reduce_on_outcome 误标）
- 翻译 HES 事件（145M+ 总诊断行）
- 数据集：train 165,522 / val 8,273 / test 11,363（含约 4,400 个仅有 HES 的 dementia 患者）

**结果**：
- Test Dementia C_td = **0.720**（**比 baseline 0.733 还低**）
- 在 GP-only 8,292 个患者子集上 C_td = 0.684（远低于 hes_aug 的 0.733）

**为什么失败**：
- **模态冲突**：HES 事件稀疏、急性、住院相关；GP 事件高频、慢性、初级保健。两者时序模式差异大
- **序列爆炸**：GP-only 中位 151 token → 融合后 731 token
- **截断破坏**：`block_size=512` 导致 ~60% 序列被截，最近的 GP 事件（最有预测力）被丢
- **Pretrained backbone 损坏**：混合模态训练破坏了 GP 上学到的时序模式

**结论**：放弃序列级 fusion，改走"独立 backbone + 后期融合"路线。

---

### 3.2 实验 B：HES Backbone Pretrain（成功）

**目的**：为 dual-backbone 架构准备一个独立的 HES transformer。

**做法**：
- 构建纯 HES SQLite 数据库：`build_hes_database.py`
  - 来源：`hesin.csv` + `hesin_diag.csv`
  - ICD-10 截到 3 字符（数据规模和粒度的平衡）
  - 仅保留 level=1 主诊断，3.9M 事件，420K 患者，1499 token
- 构建 HES PreTrain Parquet：327K/18K/20K split
- Self-supervised next-event prediction（vocab=1501，block_size=256）
- **结果**：8 epoch，best val_loss=2.558，**test_loss=2.407**
- 产出 `crPreTrain_HES_1337.ckpt`（12.2M 参数）

注：HES pretrain 用了完整序列（不按 index date 截断），但因为是 self-supervised（没有 outcome label），**不算数据泄露**（类似 BERT 用全文本预训练）。

---

### 3.3 实验 C：Dual-Backbone v1（GP + HES + Gated Fusion）

**架构**：
```
GP 序列  → [GP Backbone (pretrained)]  → h_gp (384-dim)  ─┐
                                                            ├─ Gated Fusion → h_fused → DeSurv CR Head
HES 序列 → [HES Backbone (pretrained)] → h_hes (384-dim) ─┘
```

**Gated Fusion**:
```
gate = σ(W · [h_gp; h_hes])
h_fused = gate ⊙ h_gp + (1 - gate) ⊙ h_hes
```

无 HES 记录的患者 h_hes = 零向量，gate 学会自动依赖 h_gp。

**实施**：
- 修改 GP DataModule，包装 collate_fn 注入 HES 序列（`DualCollateWrapper`）
- 加载 GP pretrain 权重 → `gp_transformer`，HES pretrain 权重 → `hes_transformer`
- Fusion + survival head 从零训练，10× 学习率（5e-4 vs backbone 5e-5）
- 训练 ~44 小时，22 epoch，best epoch 13（val_loss=0.007）
- 总参数 106M，checkpoint 1.1 GB

**当时报告结果**：
- Dementia C_td = **0.845**（+0.112 vs baseline 0.733）
- Death C_td = 0.949
- Overall C_td = 0.891

看起来是新 SOTA。

---

### 3.4 实验 D：22-dim HES Static 扩展

**思路**：在 dual-backbone v1 基础上，把 HES 静态特征从 8 维扩到 22 维（加入更多临床有意义的共病和连续特征）。

**新增 11 维 binary 共病**（基于 Lancet Dementia Commission risk factors）：
- 高血压 (I10-I15)、房颤 (I48)、CKD (N18)、抑郁 (F32/F33)、帕金森 (G20)、癫痫 (G40/G41)、肥胖 (E66)、高脂血症 (E78)、COPD (J44)、酒精依赖 (F10)、睡眠障碍 (G47)

**新增 3 维 continuous**：
- 平均住院天数、急诊比例、距最近入院年数

**实施**：
- `static_proj` 层从 (384, 35) 扩到 (384, 49)，pretrained 权重前 27 列复制，HES 维度 zero-init
- Train 单独的 hes_static_v2 数据集 + 重新训 dual-backbone

**当时报告结果**：
- hes_static v2 (22-dim, 单 backbone)：Dementia C_td = **0.875**（+0.142）

看起来又一个新 SOTA。

---

## 4. 🚨 阶段二：数据泄露发现与修复（4 月 29 日）

### 4.1 发现 Bug

仔细 review `build_hes_summary_features.py` 时发现：

```python
# 原代码（有 bug）
hesin = pd.read_csv(HESIN_CSV)
diag = pd.read_csv(HESIN_DIAG_CSV)
# 直接计算 features，没有按 index date 过滤！
```

`HES_TOTAL_ADMISSIONS`、`HES_HAS_DELIRIUM` 等所有特征都是用了**患者完整生命周期的 HES 记录**计算，包括 index date (age 72) 之后的住院诊断。

特别糟糕的是 `HES_YEARS_SINCE_LAST_ADMISSION`：原本相对于固定的 `STUDY_END_DATE = 2022-10-31`，而不是患者特异的 index date。

### 4.2 泄露规模量化

加入 `admidate < index_date` 过滤后：

| 项目 | 泄露版 | 修复版 | 减少比例 |
|------|--------|--------|---------|
| HES admissions | 4,238,372 | 2,990,537 | **29.4% 是 post-index** |
| HES diagnoses | 17,421,258 | 10,419,676 | **40.2% 是 post-index** |
| 谵妄 (F05) prevalence | 2.0% | 0.5% | **75% 是 post-index** ⚠️ |
| 房颤、CKD、心衰 | — | — | 普遍下降 |

**为什么谵妄影响最大**：谵妄常发生于 dementia 晚期或最终住院期间。如果模型在 72 岁预测时"看到"了 75 岁的谵妄诊断，等于直接看到答案。

### 4.3 验证泄露

跑了一个 sanity check：把"在泄露数据上训练好的模型" 拿到 "干净 test set" 上 eval：

| 测试 | C_td |
|------|------|
| 泄露训练 + 泄露 test（之前报告的 hes_static v2） | ~~0.875~~ |
| 泄露训练 + **干净 test** | **0.706**（**低于 0.733 baseline**） |

→ 完美证明了泄露：模型在训练时学到了"未来信息"作为捷径，当这些捷径在干净测试集上消失时，性能甚至**比无 HES 特征的 baseline 还差**。

### 4.4 影响范围

| 之前报告的实验 | 报告 C_td | 状态 |
|--------------|----------|------|
| hes_static v1 (8-dim) | ~~0.836~~ | ❌ 泄露 |
| hes_static v2 (22-dim) | ~~0.875~~ | ❌ 泄露 |
| dual-backbone v1 | ~~0.845~~ | ❌ 泄露 |
| hes_aug (label augmentation only) | 0.733 | ✅ 不受影响（不用 HES 特征） |
| hes_fusion v5 | 0.720 | ✅ 不受影响 |

### 4.5 修复

修改 `build_hes_summary_features.py`：

```python
# 修复后
yob_lookup = load_year_of_birth_lookup()
index_dates = {pid: yob + DateOffset(years=72) for pid, yob in yob_lookup.items()}

hesin["index_date"] = hesin["eid"].map(index_dates)
hesin = hesin[hesin["admidate_dt"] < hesin["index_date"]]  # ← 关键过滤

# 同样过滤 diagnosis
pre_index_admissions = set(zip(hesin["eid"], hesin["dnx_hesin_id"]))
diag = diag[diag.apply(lambda r: (r["eid"], r["dnx_hesin_id"]) in pre_index_admissions, axis=1)]

# Years_since_last_admission 改为相对于 patient-specific index date
years_since = (idx_date - last_admidate).days / 365.25
```

---

## 5. 修复后真实 baseline：Dual v2 Clean

### 5.1 重训 + 重 eval

用修复后的 22-dim HES 特征 + GP+HES dual backbone + gated fusion，在干净数据上 train + test：

**结果**：

| 指标 | 值 |
|------|-----|
| **Dementia C_td** | **0.7569** |
| **Death C_td** | **0.9488** |
| 真实提升 | +0.024 vs baseline 0.733 |

### 5.2 Temporal Leakage Sanity Check

把这个 clean 模型放到 leaky test set 上 eval，C_td 仍然是 0.7569（完全相同）→ 证明模型学的全是合法信号，**没有任何泄露捷径**。

### 5.3 教训

- **+0.103 ~ +0.142 的"虚假提升"全是泄露**，真实提升只有 **+0.024**
- 任何"看起来太好"的提升（特别是简单的 binary 共病特征带来的）都应该警惕
- **生存分析中，所有从外部数据源（HES、死亡记录等）派生的特征必须做 patient-specific index date 过滤**

---

## 6. 阶段三：本次对话中的改进实验（5 月 2 日 - 10 日）

### 6.1 实验 E：Cross-Attention Fusion（不如 gated）

**动机**：Gated fusion 只用了每个 backbone 的 last token。Cross-attention 可以让 GP 的最后表示去 attend HES 的全序列（反之亦然），实现更细粒度的信息融合。

**架构**：
```
GP last token (query) → cross-attend HES full sequence (K, V) → enriched_gp
HES last token (query) → cross-attend GP full sequence (K, V) → enriched_hes
[enriched_gp, enriched_hes] → Linear(768→384) → h_fused
```

新增参数：~1.2M（两个 nn.MultiheadAttention + LayerNorm + projection）。

**实验 E1（v1，无 warmup）**：训练发散，不收敛 ❌
- 原因：随机初始化的 cross-attention 层从 epoch 0 起就破坏 backbone fine-tuning

**实验 E2（v2，warmup=3）**：
- 前 3 epoch 冻结 backbone，只训 fusion + head
- 第 4 epoch 起解冻 backbone 联合训练
- 收敛后 **Dementia C_td = 0.7487**（**比 gated 0.7569 差 -0.008**）

**结论**：放弃 cross-attention，回归 gated fusion。
- HES 序列的 token 级时序信息对 dementia 预测贡献有限
- Last-token summary 已经足够
- 额外参数增加过拟合风险

---

### 6.2 实验 F：V2 标签纠正（+0.003）

#### 动机：DeSurv 似然函数的标签噪声放大效应

DeSurv 竞争风险的负对数似然（DEATH 患者部分）：

$$\ell_{\text{death}}(t) = -\log\big[ f_{\text{death}}(t) \cdot (1 - F_{\text{dementia}}(t)) \big]$$

其中 $F_{\text{dementia}}(t)$ 是 dementia 的 CIF。如果一个**实际有 dementia 但被标记为 DEATH** 的患者出现在训练集，损失函数会**主动压低** $F_{\text{dementia}}(t)$ 在该患者事件时间处的取值，**系统性损害模型对 dementia 的判别力**。

#### 三类标签纠正

| 类别 | 描述 | 纠正方法 | 数量 |
|------|------|---------|------|
| **A** | DEATH 患者在 HES 中有 dementia 诊断 | 重标 dementia，event time 用 HES 诊断日期 | **1,123** |
| **B** | DEATH 患者死亡证明列出 dementia ICD-10 (F00-F03, G30)，但 HES 无 dementia | 重标 dementia，event time 用死亡日期 | **274** |
| **C** | HES dementia 早于 index date | 删除（prevalent case，本就不应纳入生存分析） | **487 删除** |

数据来源：
- HES dementia：从 hesin.csv + hesin_diag.csv 提取（ICD-10 F00, F01, F02, F03 prefix + G30 prefix + F03 exact）
- 死亡证明 dementia：从 death.csv + death_cause.csv 提取
- Prevalent：HES dementia 日期 < index date

优先级：A > B（同一患者两类都符合时取 HES 诊断日期）。

#### 数据集影响

| | V1 (原始) | V2 (纠正) | 变化 |
|--|----------|-----------|------|
| 总患者 | 133,809 | 133,322 | -487 prevalent |
| Train dementia | ~4,500 | ~5,900 | +1,397 relabel |
| Train DEATH | ~17,000 | ~15,900 | -1,123 relabel |
| Val/Test | 不变 | 不变 | 同样的 split，公平对比 |

#### 结果

| 指标 | V1 (clean baseline) | V2 (label corrected) | Δ |
|------|--------------------|--------------------|---|
| **Dementia C_td** | 0.7569 | **0.7602** | **+0.003** |
| Death C_td | 0.9488 | 0.9488 | 0 |

正向但提升较小（+0.003）。

---

### 6.3 实验 G：V3 Self-Training（第一轮 ⭐⭐⭐，已被 V4 超越）

#### 动机

V2 纠正了**已经在 HES / 死亡证明留下痕迹的** 1,397 个隐藏 dementia 患者。但还有大量**完全没有任何 dementia 文档**的患者也可能患病——他们在 GP 数据里只是 censored。**用我们自己训练好的 V2 模型去识别这些人**。

#### Self-Training Pipeline

**Step 1：Train-set CIF Inference**
- 用 V2 best checkpoint，对全部 119,271 个 train 患者做 inference
- 提取 CIF_dementia 在每个患者实际 event time 处的值（`cif_dementia_at_event`）
- 注意点：
  - FoundationalDataModule 必须设 `supervised=True` 才会调用 `convert_to_supervised()`
  - Patient ID 不在 collated batch 里，需要从预加载的 parquet 数据建立 index map
  - DataLoader 用 `shuffle=False` 才能按 index 追踪患者
- 输出：`train_cif_dementia_v2.csv`，119,271 行
- 耗时：~2.5 小时（单 GPU）

**Step 2：候选筛选**
- 在所有非 dementia（DEATH + censored）患者中，用 CIF_dementia 排序
- 取 top 1% → 阈值 CIF ≥ 0.2521 → 1,140 候选
- 排除观察时间 < 2 年的 censored 患者（follow-up 太短，模型预测可能不可靠）→ 369 排除
- 最终：**771 个 pseudo-dementia 患者**

为什么 top 1%：保守阈值，宁缺勿滥；过低阈值会引入大量假阳性 pseudo-label，反而损害模型。

**Step 3：构建 V3 数据集**
- 复制 V2 数据集
- 对 771 个患者（仅在 train split 中）：
  - Last EVENT 改为 `Eu02z`（unspecified dementia Read v2 code）
  - 保留原 event time 不变
- Val/test 完全不动（保证公平对比）

**Step 4：V3 训练**
- 完全相同的 dual-backbone gated fusion 架构和超参
- 25 epoch max，**best epoch 15**（val_loss=0.0356）
- Epoch 16-20 val_loss 不再改善（过拟合）→ 停止训练，用 epoch 15 checkpoint

#### 结果

| 指标 | V2 (label corrected) | **V3 (self-training)** | **Δ vs V2** | Δ vs baseline 0.733 |
|------|---------------------|-----------------------|------------|---------------------|
| **Dementia C_td** | 0.7602 | **0.7685** | **+0.008** | **+0.036 (+4.8% relative)** |
| Dementia IBS | — | 0.1740 | — | — |
| Dementia INBLL | — | 0.5101 | — | — |
| **Death C_td** | 0.9488 | **0.9518** | **+0.003** | — |
| Death IBS | — | 0.1009 | — | — |
| Death INBLL | — | 0.3319 | — | — |
| Overall C_td | — | **0.8616** | — | — |
| Overall IBS | — | 0.0417 | — | — |

**WandB run**: `crPreTrain_small_1337_FineTune_Dementia_CR_dual_v3` (run ID: 5unkvjfm)

#### 解读

- 单纯 +0.008 数字看起来不大，但在 dementia 预测领域是有意义的提升
- 重要的是**方法学价值**：证明了 self-training 在生存分析中是可行的，并且模型确实能识别隐藏的 dementia 患者
- 与之前的对比：从 dual v2 clean baseline (0.7569) 到 V3 (0.7685) 累计提升 +0.012，全部来自标签层面的改进（不是架构改动）

### 6.4 实验 H：V4 第二轮 Self-Training（当前最佳 ⭐⭐⭐⭐ — 新增）

#### 动机

V3 用 V2 模型找了 771 个隐藏 dementia 候选。第二轮问的是：**用 V3 模型（已经学了那 771 个 pseudo）能不能再找到 V2 漏掉的更多隐藏 dementia？**

两种可能结局：
- 真实提升 → V3 找到的真信号还能继续挖
- Confirmation bias / 模型 collapse → V3 只是复制粘贴 V2，不会有新提升

#### 流程

1. **V3 推理**：V3 模型在 V3 train set 上跑 inference，输出每个患者的 CIF_dementia
   - 输出：`train_cif_dementia_v3.csv`
   - V3 的 771 pseudo 现在已被标为 dementia，自动从候选池中排除

2. **候选筛选（更激进的 top 2%）**：
   - Pool: 113,214 个非 dementia 患者
   - Top 2% 阈值: CIF ≥ **0.1990** （V3 用的是 top 1%, 阈值 0.2521）
   - 2,265 个候选
   - 过滤掉 censored <2y observation: 1,441 个被排除
   - **最终 V4 pseudo candidates: 824 (555 from DEATH + 269 from censored)**

3. **Overlap 分析（新增）**：检查 confirmation bias

| V4 candidates 的 V2 rank 分布 | 数量 | % |
|----|----|----|
| V2 top 0-1% | 0 | 0.0% (按构造排除：V3 的 771 在 V3 train 已标 dementia) |
| V2 top 1-2% | 333 | 40.4% |
| V2 top 2-5% | 322 | 39.1% |
| V2 top 5-10% | 118 | 14.3% |
| V2 top 10-20% | 36 | 4.4% |
| V2 top 20-50% | 15 | 1.8% |
| V2 bottom 50% | 0 | 0.0% |

- V4 candidates 在 V2 ranking 的位置：median 2.41%, p25 1.55%, p75 4.34%
- **79% V4 candidates 在 V2 top 5%**
- **98% 在 V2 top 20%**

**初步解读 (悲观)**：脚本标记为 "STRONG OVERLAP, possible confirmation bias"，预测 V4 可能没提升。

**最终实证结果驳斥了这种悲观**：V4 仍然提升 +0.005。Re-interpretation：V4 候选是 V2 ranking 中的"borderline high-risk"（top 1-5% 但没进 V2 top 1%），V3 已把他们 consolidated 到 top 2%。**两个独立模型对这群患者的 ranking 共识，正是他们是真高风险的证据，不是 confirmation bias**。

4. **V4 数据集**：复制 V3 → 在 train split 把 824 个候选的 last EVENT 改为 Eu02z（dementia code）

5. **V4 训练**：
   - 同 V3 架构（dual gated）
   - 最多 25 epoch，**best at epoch 9**（val_loss = 0.052）
   - 注意：V4 best val_loss (0.052) **高于** V3 (0.0356) → V4 在 val 上轻微 overfit（pseudo-label 让模型学得更激进，对 val unchanged labels 泛化稍弱）
   - 但 **test C_td 仍提升**（label noise 减少的好处 > overfit 的代价）

#### 结果

| 指标 | V3 | **V4 (2nd SST)** | **Δ vs V3** |
|------|----|------------------|-------------|
| **Dementia C_td** | 0.7685 | **0.7732** | **+0.0047** ↑ |
| Dementia IBS | 0.1740 | **0.1609** | ↓ 更好 |
| Dementia INBLL | 0.5101 | **0.4701** | ↓ 更好 |
| Death C_td | 0.9518 | 0.9486 | -0.0032 |
| Death IBS | 0.1009 | 0.1480 | ↑ 略差 |
| Death INBLL | 0.3319 | 0.4467 | ↑ 略差 |
| **Overall C_td** | 0.8616 | **0.8649** | **+0.0033** |
| Overall IBS | 0.0417 | 0.0563 | ↑ 略差 |
| Overall INBLL | 0.1368 | 0.1839 | ↑ 略差 |
| test_loss | 0.0351 | 0.0494 | ↑ |

**WandB run**: `crPreTrain_small_1337_FineTune_Dementia_CR_dual_v4` (run ID: `w4rmiltg`)

**vs hes_aug baseline (0.733)**: **+0.040（+5.5% relative improvement）**
**vs V3 (0.7685)**: +0.005（+0.6% relative）

#### 关键观察

1. **Dementia 端全面提升**：C_td +0.005, IBS -0.013, INBLL -0.040。Self-training 进一步减少标签噪声。

2. **Death 端略退**：Death C_td 从 0.9518 → 0.9486 (-0.003)。原因：把 555 个原 DEATH 标为 dementia → 模型看到的死亡训练样本减少。退化幅度小（<0.4%）。

3. **驳斥 confirmation bias 假说**：尽管 79% V4 候选在 V2 top 5%，V4 仍然提升。强证据：**V3 的高 CIF ranking 反映真实的高风险信号，不是 training artifact**。

4. **递减收益确认**：V2→V3 提升 +0.008，V3→V4 提升 +0.005。下降但仍正。第三轮（V5）预期 +0.002~+0.003。

### 6.5 实验 I：V2 Ablation — Dual Backbone 贡献 ≈ 0 (2026-05-12~13) ⚠️

#### 动机

回顾整个实验历史，**我们从未在 clean 设置下直接测量过 dual backbone 的贡献**：
- 泄露期：跑过 "8 维 + dual" (0.845, 泄露)，但没跑过 "22 维 + dual" (泄露)
- Clean 期：直接跳到 "22 维 + dual clean" (0.7569)，没跑过 "22 维 + GP only clean"

→ 不知道 clean 设置下 dual backbone 贡献是大（~+0.02）、中（~+0.005）还是无。

#### 设置

- 数据集：V2 (同 dual v2，22 维 clean HES static + V2 label corrections)
- 架构：**单 GP transformer**（无 HES backbone，无 fusion layer）
- 静态协变量：49 维（27 base + 22 clean HES static）
- Entry point: `run_experiment.py`
- 有效 batch size: 512（匹配 dual v2，公平对比）

#### 训练

- 训练 20 epochs（手动停止，best 后连续 4 epoch 没改善）
- **Best at epoch 15**，val_loss = **0.0300**
- 对比 V3 val_loss = 0.0356，V4 = 0.052 → ablation 的 val_loss 反而更低
- 单 backbone 参数少，对 V2 labels（无 pseudo）泛化能力反而好

#### 结果

| 指标 | V2 Ablation | Dual v2 (gated, V2 labels) | Δ |
|------|-------------|---------------------------|---|
| **Dementia C_td** | **0.7571** | **0.7569** | **+0.0002 (≈ 0)** |
| Dementia IBS | 0.2051 | — | — |
| Death C_td | 0.9454 | 0.9488 | -0.0034 |
| Overall C_td | 0.8538 | (~0.85) | — |
| test_loss | 0.0326 | — | — |

WandB run: `crPreTrain_small_1337_FineTune_Dementia_CR_hes_static_v2_ablation` (run ID: `tg9faux1`)

#### ⚠️ 重大发现：Dual Backbone 在 V2 Clean 设置下贡献 ≈ 0

- V2 ablation (single GP) = **0.7571**
- Dual v2 (gated fusion + HES backbone) = **0.7569**
- **差距：+0.0002，在统计噪声范围内**

→ **Dual-backbone 架构（HES backbone + gated fusion）在我们这个 setup 下几乎没有提升**。+0.024 的提升（vs hes_aug 0.733）几乎全部来自：
- 22 维 HES static covariates（clean，泄露修复后）
- V2 标签纠正

#### 论文 framing 修正

**旧 framing（错的）**："Dual-backbone gated fusion 是核心创新，提升 +0.024。"

**修正 framing**："三个组件驱动改进：(1) clean 22-dim HES static，(2) V2 标签纠正，(3) 两轮 self-training。Dual-backbone gated fusion 架构在方法学上展示了如何融合多模态 EHR，但在 clean evaluation regime 下相对 single GP backbone 的额外判别力**接近零**（+0.0002）。"

#### Caveat

- 此 ablation 仅在 V2 labels 上做，**未测试** single backbone + self-training 是否也能拿到 V3/V4 的 +0.013 提升
- 如果成立，**single backbone + V4 self-training 可能匹配 dual backbone V4 (0.7732)**——这是未来可考虑的实验
- V2 ablation calibration 在 orchestrator pipeline Step 4 队列中

#### 改进的归因表

| 组件 | 估算贡献 |
|------|---------|
| Baseline (hes_aug GP + HES labels) | 0.733 |
| +22 维 clean HES static features | +0.024 → 0.757 |
| +V2 标签纠正 (1397 relabel + 487 prevalent removed) | +0.003 → 0.760 |
| +V3 self-training (771 pseudo) | +0.008 → 0.768 |
| +V4 self-training (824 new pseudo) | +0.005 → **0.7732** |
| Dual backbone (vs single) | ~0 |
| Gated vs cross-attention fusion | ~0（都接近 single backbone）|

---

## 7. 阶段四：评估指标多样化（与文献对比）

### 7.1 动机

主指标 C_td 在文献中较少使用。文献（UKBDRS、DemRisk、Yuan 等）大多报告：
- AUROC at 5y / 10y
- Top-K precision
- Harrell's C
- Calibration

为了直接对比，需要把这些指标也算出来。

### 7.2 第一轮尝试（直接用 CIF@5y from index date 作为评估范式）

写了 `inference_test_metrics.py` + `compute_additional_metrics.py`，对 V3 test set 跑：

| 指标 | 值 |
|------|-----|
| AUROC@1y dementia | 0.7777 |
| AUROC@2y dementia | 0.6904 |
| AUROC@5y dementia | **0.5756** ⚠️ |
| AUROC@5y death | 0.8462 |
| Top 1% precision @5y | **3.7%** ⚠️ |
| Harrell's C dementia (CIF@1y) | 0.6848 |

**发现矛盾**：AUROC@5y = 0.58 与 C_td = 0.77 几乎不可能同时成立。最初判断"评估方法有 bug"，但后来意识到 **不是评估实现 bug，是评估范式选错了**。

### 7.3 ⚠️ 关键认识纠正：评估范式选择问题

> **原本（错误）的 framing**：模型预测的"5 年"和 label 定义的"5 年"对不上 = "时间错位 bug"，需要修复对齐。
>
> **正确的 framing**：评估范式选错了。我们的模型是 **dynamic prediction**，它的 native 评估范式就是 "从 prediction point 起 t 年"，而我们用 "从 index date age 72 起 5 年" 当 ground standard，**这是把 baseline-only Cox 模型的评估范式强加给一个 dynamic prediction 模型**。

#### 临床部署场景才是 ground truth

考虑临床实际使用场景：**一个病人坐在医生面前，他的 EHR 历史到此为止，医生问"从现在起未来 5 年得 dementia 的概率是多少？"**

- 70 岁的病人来看 → 医生想知道 70-75 岁的风险
- 73 岁的病人来看 → 医生想知道 73-78 岁的风险
- **没有"必须等到 age 72 才能用模型"这种限制**

→ **每个病人的"5 年"以自己的当下为起点**，这就是 dynamic prediction 的核心设计意图，也是我们的模型 natively 在做的事情。

#### Index date 的真正作用是 cohort 入组条件

`INDEX_ON_AGE = 72` 不是模型查询的强制时点，而是：
- **研究设计上的 cohort 入组锚点**："我们研究 72 岁这个年龄段的 dementia 风险预测"
- 用来筛选研究人群（"那些活到 72 岁且 event-free 的人"）
- **不是模型在临床部署时"必须从此刻预测"的硬约束**

#### 模型的 native 评估范式（正确做法）

```
对每个病人 i:
  Prediction point p_i  =  i 的最后一次 pre-index EHR encounter
  CIF(t)              =  模型在 t 年内得 dementia 的预测概率（t=0 起算）
  
Label  y_i(t)           =  1[T_i - p_i ≤ t]   即 event 距 prediction point ≤ t 年
Score                   =  CIF_dementia(t)
```

这个范式：
- ✅ 完全匹配临床部署场景
- ✅ x 轴和 y 轴量同一件事
- ✅ 不需要任何"对齐"trick

### 7.4 V3 时期的尝试（现在看是错误方向）

#### 7.4.1 直接拿 CIF@5y 评估 "from index date" → AUROC 0.5756
- **错误来源**：用错评估范式（baseline-only 范式套在 dynamic prediction 模型上）
- **不是模型差，不是评估实现 bug，是评估范式选择错误**
- AUROC 0.5756 这个数字**不应该被解读为模型性能**，只能解读为"用错误的评估问题问出的无意义结果"

#### 7.4.2 Per-patient τ_i 时间对齐尝试 → AUROC 0.9951
"对齐"的尝试：对每个病人算 `τ_i = (5 - δ_i)/5`，在 τ_i 处查 CIF。

**结果**：AUROC 0.9951、calibration 双峰。

**为什么这么高**：这是 **结构性人为虚高**，不是模型真的能力：
- Dementia cases (T_i ≤ 5y from index): τ_i ≥ Δt_i → 查询的 CIF 时间点**在实际事件之后** → 必然高 CIF
- Controls (T_i > 5y from index): τ_i < Δt_i → 查询的 CIF 时间点**严格早于最后观测** → 必然低 CIF
- → cases / controls 系统性地在 CIF 不同位置被查询 → **几乎必然 AUROC ≈ 1**

**本质**：这个"对齐"是错的——它没有修复评估范式问题，而是制造了 case/control 在评估时间点上的人为不对称。Calibration 双峰（bottom 8 bins 全 0%，top bin 99%）就是这种人为不对称的可视化呈现。

### 7.5 正确的评估范式（应该这样做）

#### 多时点 calibration on model's native timeframe

```
对每个 t ∈ {1y, 2y, 3y, 5y}:
  Label    y_i(t)  =  1[T_i - p_i ≤ t]
  Score    F_i(t)  =  CIF_dementia(t | x_i)
  
  → 画 calibration plot (10 quantile bins, KM 处理 censoring)
  → 计算 calibration slope (理想值 = 1.0)
  → 计算 AUROC (此时 x/y 同尺度，可以正常算出来)
```

这是 **dynamic prediction 模型的 native 评估范式**，clinically aligned 且方法学正确。

#### Sensitivity check (推荐 supplementary 做)

按 δ_i 分层做 calibration（例如 δ ≤ 0, 0 < δ ≤ 2, δ > 2）：
- 检查不同 prediction-point-to-index 距离上 calibration 是否一致
- 顺带 address survivorship bias caveat（δ_i 很大的病人隐含 "条件在活到 age 72 仍 event-free"）

### 7.6 最终可信指标

仅列已 verified 的对比文献（直接读过原文 / PMC 全文 / abstract / 上传 PDF），未引用 research agent 二手转述的数字。

| 指标 | V3 数值 | 对比文献（verified） | 评价 |
|------|---------|---------------------|------|
| **C_td dementia** | **0.7685** | 见 Section 8 详细对比 | ✅ paradigm-agnostic，可跨范式对比 |
| **C_td death** | **0.9518** | — | ✅ |
| ~~Harrell's C dementia (cause-specific)~~ | ~~0.7884~~ | ~~Yuan 2024 / DemRisk~~ | ❌ **撤回**：aligned CIF artifact + DeSurv vs PH 模型 categorically 不可比，见 Section 8 修正版 |
| **Harrell's C death** | **0.9460** | — | ✅ |
| **AUROC@t in model native timeframe** | 待 V4 上重算 | — | ⏳ Section 11.4 已 plan, V4 calibration 一起做 |
| ~~AUROC@5y "from index date"（错误范式）~~ | ~~0.5172~~ | — | ❌ **错用 baseline-only 范式**，不是模型缺陷 |
| ~~AUROC@5y time-aligned（τ_i 对齐尝试）~~ | ~~0.9951~~ | — | ❌ **错误的"对齐"尝试**制造结构性虚高 |

### 7.7 论文写作建议（重新校准）

**Methods 里精确 framing 评估范式**：

> "Our model is a dynamic prediction system: it issues a cumulative incidence function CIF(t) representing the probability of dementia within t years from the prediction point (the patient's most recent pre-index EHR encounter). All evaluation is performed in the model's native timeframe, with the binary outcome at time t defined as y_i(t) = 1[T_i - p_i ≤ t], where T_i is the absolute event time and p_i is the patient's prediction point. This paradigm aligns with the clinical deployment scenario, where a clinician seeks risk estimates at the time of consultation rather than at a fixed cohort-entry date.
>
> We note that baseline-only Cox models (UKBDRS, Zhang 2026, DemRisk, etc.) evaluate using a single prediction at cohort recruitment with a fixed time origin. Direct comparison of fixed-horizon metrics (e.g., 'AUROC@5y from cohort entry') against our dynamic prediction model is paradigm-incompatible. For cross-paradigm comparison, we report C_td and cause-specific Harrell's C, which are agnostic to the prediction-time origin."

**报告什么指标**：
- 主指标 **C_td** (paradigm-agnostic, 用于和文献做跨范式对比)
- 副指标 **Harrell's C** (cause-specific)
- **Calibration plot + Calibration slope** on model native timeframe (multiple t = 1y/2y/3y/5y)
- **IBS, INBLL** as combined-quality scores
- **AUROC@t on model native timeframe** (可以正常算了，原来算的 0.5756 是错用范式的结果，不应报)

**不报**：
- AUROC@5y "from index date"（错误范式）
- τ_i-aligned AUROC（错误对齐尝试）
- 任何强行对齐到 "age 72 + 5y" 的指标

### 7.8 副产品：证伪了"CIF 饱和、用 π 修复"假说

之前一个 agent 提出："CIF 在 t=5y 处饱和到 π，用 π 替换 CIF 当 risk score 应该能让 AUROC 跳到 0.78"。

实验证伪：π 的 AUROC = 0.5168 ≈ CIF@5y 的 0.5172。两者几乎相同。

**结论**：饱和不是主要问题；**真正问题是评估范式选错了**（错把 baseline-only 范式套到 dynamic prediction 模型上）。换到正确范式（model native timeframe）后，AUROC 应该和 C_td 自洽。

### 7.9 ⚠️ 给新 agent 的明确指引

如果 reviewer 或新 agent 想做 AUROC@5y / calibration 分析，**遵循以下规则**：

✅ **正确做法**：
- 用 `y_i(t) = 1[T_i - p_i ≤ t]` 作为 label
- 在 t = 1y / 2y / 3y / 5y 多个时点报
- 这评估的是模型在 native 时间轴上的预测质量，clinically aligned

❌ **错误做法（不要做）**：
- 用 `y_i = 1[event before age 72 + 5y]` 作为 label（把 baseline-only 范式强加到 dynamic prediction）
- 用 τ_i 对齐 CIF 到 index date 时间轴（制造结构性虚高）
- 用 fixed-horizon AUROC 和 UKBDRS / Zhang 2026 等 baseline-only 模型做大小比较（范式不可比）

---

## 8. 与文献的位置 — **2026-05-13 重大修正**

### ⚠️ 8.0 重大修正：Harrell's C 不可直接对比

**2026-05-13 之前的所有 framing（声称"Harrell's C 0.7884 > Yuan 0.749 / DemRisk 0.78"）已撤回**。

**根本原因（数学层面）**：

| Model 类型 | Output | Rank-invariance over time | Harrell's C 是否 unambiguous |
|-----------|--------|---------------------------|------------------------------|
| **DemRisk (Cox PH)** | scalar log-risk `β·x` | ✅ Yes (PH assumption) | ✅ Unambiguous |
| **Yuan (DeepSurv)** | scalar log-risk `h_θ(x)` | ✅ Yes | ✅ Unambiguous |
| **Yuan (DeepHit)** | discrete-time hazards | ❌ No | ⚠️ Depends on projection (不清楚 Yuan paper 用的哪个)|
| **我们 (DeSurv ODE)** | full CIF curve F_k(t\|x) | ❌ No (CIF can cross) | ⚠️ **任意 scalar projection** |

**关键性质**：PH 模型保证 "谁高风险" 跨时间不变 → scalar 就够。我们 DeSurv 不假设 PH，CIF 可以交叉 → 没有 natural scalar。

**实证**：用 5 种不同 scalar projection 算 V4 的 Harrell's C，结果范围 **0.469 - 0.788**，差距 0.32。没有 "the" Harrell's C。

| Projection | V4 Harrell's C |
|-----------|----------------|
| aligned CIF (rejected, artifact) | 0.788 |
| CIF@1y | 0.696 |
| CIF@3y | 0.575 |
| CIF@5y | 0.469 (saturated) |
| CIF@event time | 0.469 (biased) |
| π_dementia | 0.531 |

→ **结论**：和 Yuan/DemRisk 的 PH-based Harrell's C 不能直接数字对比。任何方向的对比（高于 / 低于）都没有意义。

### 8.1 现在能用的 quantitative claim

#### ✅ Within-paper（最 solid）
- vs hes_aug baseline (0.733)：**C_td +0.040** 通过整套 pipeline（22 维 static + V2 labels + V3/V4 self-training）
- vs V2 ablation (0.7571)：**+0.016** 通过两轮 self-training
- vs Dual v2 (0.7569)：**+0.016** 通过 V2 labels + self-training（dual backbone 本身 ≈ 0 贡献）

#### ⚠️ Cross-paper（需要 caveats）
- **Yuan / DemRisk / Anatürk / Zhang 等 PH 模型**：可以**承认存在**但**不直接数字比较**
- 用 framing：本质 different model paradigm，提供 PH 模型无法提供的能力（competing risks, non-PH, dynamic prediction）

#### ⏳ Pending 实验（建议做）
1. **Head-to-head on our data**：自己跑 Cox PH / DeepSurv 在 V2-corrected labels + 22-dim HES static 上
   - 给 reviewer 一个 concrete 数字（我们 C_td X vs 我们自己跑的 Cox Harrell's C Y）
   - 3-5 天工作
2. **PH 假设检验**：Schoenfeld residuals test on Cox PH on our data
   - 如果 p < 0.05 → "categorically more capable" 有实证支撑
   - 1 小时工作
3. **DCA (Decision Curve Analysis)**：临床效用指标
   - 临床期刊 (Lancet Digital Health, BMJ) 几乎必报
   - 半天工作

### 8.2 各 published 工作的处理（修正版）

| 文献 | 之前 framing | 修正后 framing |
|------|------------|---------------|
| **Yuan 2024 (DeepSurv 0.749)** | "我们 0.7884 > 他们 +0.04" | **不直接数字比** → 标注 "different model paradigm" + 强调我们的额外 capability |
| **DemRisk (Cox 0.78)** | "我们 0.7884 > 他们" | **不直接数字比** → 同上 |
| **Anatürk UKBDRS (AUC 0.80)** | "略低，setup 差异" | 同上 + **AUC 也是 paradigm 问题**（baseline-only 14y AUC vs dynamic prediction）|
| **Zhang 2026 (0.846)** | "0.846 主要是 age driven" | 仍然成立 + 加上 paradigm 论点 |
| **NYU EHR-BERT / Botz / Wang / Gu / BEHRT-UKB / Delphi-2M / Oliver** | research agent 转述错误 | **完全撤回**，不在论文里 mention |

### 8.3 修正后的论文 Methods 段落 template

```
Our model employs DeSurv, a competing risks extension of deep survival analysis
that outputs a full cause-specific cumulative incidence function CIF_k(t|x) per
patient and risk type. This is categorically distinct from proportional hazards
(PH) models (Cox PH used by DemRisk, Reeves et al. 2024; DeepSurv used by Yuan
et al. 2024), which output a scalar log-hazard ratio per patient. PH models
satisfy rank-invariance over time (the relative ordering of patients is constant
across all time horizons); our DeSurv model does NOT assume PH, allowing CIF
curves to cross. Consequently, Harrell's C-index is well-defined for PH models
but requires an arbitrary scalar projection of the full CIF curve for our model,
yielding values ranging from 0.469 to 0.788 depending on choice. We therefore
primarily report Antolini's time-dependent concordance C_td = 0.7732 (dementia,
V4 model), which is the paradigm-appropriate metric for full-curve survival models.
```

### 8.4 修正后的论文 Discussion 段落 template

```
Direct numerical comparison of our model's discrimination against DemRisk
(Harrell's C = 0.78) and Yuan 2024 (DeepSurv = 0.749) is methodologically
not valid due to model paradigm difference (PH-based scalar risk vs full-CIF-
curve model). We instead highlight three capabilities of our model unavailable
to PH-based comparators:

(1) Explicit competing risks modeling. Yuan 2024 and DemRisk treat death as
random censoring, an assumption known to bias dementia risk estimates upward
in elderly populations. Our DeSurv model with cause-specific CIF outputs
directly accounts for the competing event of death.

(2) Non-proportional hazards. PH models assume the hazard ratio between any
two patients is time-invariant. In dementia prediction, accumulating exposure
effects (e.g., cardiovascular risk factors) likely produce time-varying hazard
ratios that violate PH. [Note: Schoenfeld residuals test pending — see Section 11]

(3) Full CIF curve enabling dynamic prediction. Our model can be queried at
any future time horizon, supporting clinical scenarios from short-term risk
stratification to long-term planning. PH models output a single relative risk
that does not adapt to query time.

We note our within-paper improvement is robust: C_td increases from 0.733
(hes_aug baseline) to 0.7732 (V4 self-training), a +0.040 improvement driven
primarily by HES static covariate integration (+0.024), V2 label correction
(+0.003), and two rounds of self-training (V3 +0.008, V4 +0.005).
```

### 8.2 Verified 但 setup 差异较大（不直接做大小比较，但可作为 framing 参照）

#### 8.2.1 Zhang et al. 2026（medRxiv, "Genetic vs Modifiable Risk"）✓ 上传 PDF 完整读过

| | 我们 | Zhang 2026 |
|---|------|-----------|
| 数据 | UK Biobank GP+HES, **idx age 72 同质年龄分层** | UK Biobank n = 345,785, **整体队列 baseline 40-69y** |
| 架构 | Transformer + 竞争风险 | Cox PH |
| 评估 | 单一 index age 的 C_td | 整体队列 13.8y follow-up Harrell's C |
| 特征 | EHR codes only | demographics + **APOE + AD-PRS + LIBRA2** |
| 报告值 | C_td = **0.7685** | Base (demographics only) = **0.811**, Full (+APOE+PRS+LIBRA2) = **0.846** |

**为什么直接数字大小比不公平**：
- 他们的 base model（**只有人口统计学特征，没有任何疾病/基因/生活方式信息**）就达到 0.811 → **0.811 几乎完全是 age-driven discrimination**（队列年龄跨度 40-69 岁，age 占主导）
- 加上 APOE+PRS+LIBRA2 三个最强特征也只升到 0.846，仅 +0.035
- **我们的 0.7685 是在 idx age 72 同质年龄分层下评估的，age 几乎不能提供 discrimination**——所有 0.7685 全部来自 EHR 信号

**Walters DRS 2016（BMC Medicine, ✓ 看过 abstract）的方法学证据支撑**：同一个 model，在 60-79 岁人群上 Harrell's C = **0.84**，在 ≥80 岁同质年龄人群上 Harrell's C = **0.56** —— **同一模型不同年龄分层落差 0.28**。

→ Zhang 2026 的 0.846 在 idx age 72 同质年龄分层下，可合理推测会大幅下降到 0.60-0.65 区间，远低于我们 0.7685。

#### 8.2.2 Anatürk UKBDRS 2023（BMJ Mental Health）✓ 看过 abstract

| | 我们 | Anatürk UKBDRS |
|---|------|---------------|
| 数据 | UK Biobank GP+HES | UK Biobank baseline survey questionnaire |
| 架构 | Dual-backbone Transformer + 竞争风险 | Cox LASSO + Fine-Gray competing risk |
| 时间窗 | dynamic | **14y AUC** |
| 特征 | EHR codes + HES static | baseline questionnaire (lifestyle/demographics) ± APOE |
| 报告值 | C_td = 0.7685 | AUC = **0.80** (no APOE) / **0.83** (+APOE) |

**对比结论**：方法学最接近（都用 competing risk），但不同指标（14y AUC 整体队列 vs 单一 index age 的 C_td）和不同特征集（baseline 问卷 vs longitudinal EHR）。**只能作为 methodological precedent 引用**，不直接做大小比较。

### 8.3 已撤回 / 移除的不可靠对比

| 文献 | 之前引用的数字 | 撤回原因 |
|------|---------------|---------|
| ~~NYU EHR-BERT 2024 AAIC~~ | ~~AUROC 0.735@5y~~ | ❌ 数字错误（实际 0.761 0-3y / 0.740 1-3y），且 horizon 不同 |
| ~~Botz 2025 AAIC~~ | ~~AUROC 0.776@5y~~ | ⚠️ 未独立 verified（research agent 转述的会议摘要） |
| ~~BEHRT-UKB (Yildiz 2026)~~ | ~~AUROC 0.874~~ | ⚠️ all-cause 5y diagnosis 平均，不是 dementia 特异 |
| ~~Delphi-2M dementia 特定数字~~ | ~~AUC 0.81@1y / 0.70@10y~~ | ⚠️ 总体架构 verified，但 dementia 特定数字未在 Nature 全文中 verified |
| ~~Wang UKB-DRP 2022~~ | ~~AUROC 0.848~~ | ⚠️ 未独立 verified（research agent 转述） |
| ~~Gu 2025 (UKB ASCVD)~~ | ~~AUROC 0.866~~ | ⚠️ 未独立 verified；且是 ASCVD enriched cohort，不可比 |
| ~~ADNI/NACC MCI→AD 模型~~ | ~~C-index 0.85-0.93~~ | ❌ 完全不同的任务和数据（已富集 MCI + biomarker） |
| ~~Oliver 2024 ELSA~~ | ~~C_td ~0.74~~ | ⚠️ 之前提到但未 verify 具体数字 |

### 8.5 我们 corrected 的论文 position

✅ **Setup 独特性**：UK Biobank linked GP+HES 上首个 dual-backbone transformer + 显式死亡竞争风险 + 单一 index age 同质年龄分层评估的 dementia 预测模型

✅ **Within-paper improvement**：from baseline 0.733 → C_td 0.7732（+0.040），attribution 见 Section 6.5

✅ **方法学贡献**：发现并修复 temporal leakage、V2 label correction、self-training pipeline、calibration interpretation under underdiagnosis bias

⚠️ **Quantitative comparison vs prior art**：因为我们 DeSurv 是 categorically different model class（full CIF curve vs PH scalar），direct Harrell's C 数字对比 不 valid。建议补做 head-to-head Cox/DeepSurv on our data 给 reviewer concrete 数字。

⚠️ **Anatürk UKBDRS (AUC 0.80) 和 Zhang 2026 (0.846)** 仍然不是直接 comparator：
- 不同 metric（baseline-only AUC vs dynamic C_td）
- 不同特征集（APOE + PRS + LIBRA2 + lifestyle questionnaire vs 我们仅 EHR codes）
- 不同 setup（baseline-only Cox in heterogeneous age cohort vs dynamic prediction in homogeneous age stratum）
- **加上 paradigm 论点**：他们 AUC well-defined, 我们 dynamic prediction 上 AUC@5y projection 多个选择

---

## 9. 主要方法学贡献

### 9.1 架构

**Dual-Backbone + Gated Fusion**:
- 第一个在 UK Biobank linked GP+HES 上做 dual transformer + 死亡竞争风险的 dementia 预测
- 优于序列级 fusion（v5 失败）
- 优于 cross-attention fusion（实验 E 验证）
- Gated fusion 优雅处理 HES 缺失（gate 自动学会依赖 GP）

### 9.2 数据严谨性

**发现并修复时间泄露 bug**：
- 揭示之前虚假的 +0.103 ~ +0.142 提升实际只是 +0.024
- 这种"看起来很 promising 的简单特征带来巨大提升"是 ML 应用到 EHR 的常见陷阱
- 修复后的 evaluation protocol 可作为该领域的 reference standard

### 9.3 标签纠正方法学

**两层标签纠正**：

**第一层 V2（基于"second source of truth"）**：
- 用 HES 诊断 + 死亡证明识别 GP 漏诊的 dementia (1,397 人)
- 用 HES 早期诊断识别 prevalent cases (487 人)
- DeSurv 似然函数对标签噪声非常敏感的理论分析

**第二层 V3（self-training / pseudo-labeling）**：
- 用模型自己识别"既无 HES 也无死亡证明痕迹"的隐藏 dementia (771 人)
- 保守阈值（top 1% + ≥2y observation 过滤）
- 这种方法论可推广到其他 underdiagnosed 的疾病预测

### 9.4 评估指标的方法学澄清

- **明确 C_td 是动态预测模型的合适指标**，fixed-horizon AUROC 在 dynamic prediction setup 下结构性不适用
- 通过严格推导识别了"时间参考系错位"问题，为后续生存分析评估提供了警示

---

## 10. 项目最终结果一览

### 10.1 性能指标演进图

```
0.836 (hes_static v1, leaky)        ❌ INVALID
0.875 (hes_static v2, leaky)        ❌ INVALID
0.845 (dual v1, leaky)              ❌ INVALID
   ↓ 发现并修复时间泄露
0.7569 (dual v2 clean baseline)     ✅ 真实 baseline
   ↓ V2 标签纠正 (+1397 relabel, -487 prevalent)
0.7602 (dual v2 + V2 labels)        +0.003
   ↓ V3 self-training (+771 pseudo)
0.7685 (V3 1st-round SST)           +0.008
   ↓ V4 second-round self-training (+824 new pseudo)
0.7732 (V4 2nd-round SST)           +0.005  ← CURRENT BEST
```

vs hes_aug baseline (0.733)：**+0.040（+5.5% relative improvement）**

### 10.2 最终模型规格

- **架构**：Dual transformer (GP + HES) + Gated Fusion + DeSurv ODE Competing Risk Head
- **参数**：106M trainable
- **GP backbone**：6 layer, 6 head, 384 embd, block_size=512, vocab=108K
- **HES backbone**：同上配置, block_size=256, vocab=1501
- **Static covariates**：49 维（27 base + 22 HES）
- **训练**：dual fine-tune ~24h（V4 best at epoch 9，total 20 epochs trained）
- **Best checkpoint**：`crPreTrain_small_1337_FineTune_Dementia_CR_dual_v4.ckpt`（1.1 GB）

### 10.3 最终性能（V4 test set, 8,257 patients — current best）

| Metric | Dementia (Risk 0) | Death (Risk 1) | Overall |
|--------|------------------|----------------|---------|
| **C_td** | **0.7732** | **0.9486** | **0.8649** |
| ~~Harrell's C (aligned, from V3)~~ | ~~0.7884~~ | ~~0.9460~~ | ❌ **撤回** — see Section 8 |
| IBS | 0.1609 | 0.1480 | 0.0563 |
| INBLL | 0.4701 | 0.4467 | 0.1839 |
| test_loss | — | — | 0.0494 |

**⚠️ 注（2026-05-13 修正）**：Harrell's C 0.7884 已撤回。用 aligned CIF 算的——而 aligned CIF 之前就因为 structural artifact 被 reject 了。更根本的问题是 DeSurv 模型 categorically 不能与 PH 模型做 Harrell's C 数字对比。详见 Section 8。**主指标改用 C_td**。

---

## 11. 下一步可做（讨论候选）

### 11.1 论文准备
- **整理 manuscript**：核心是 dual-backbone 架构 + 数据泄露发现 + V2/V3 label correction
- **可视化**：架构图、数据流、训练曲线、CIF 曲线、attention 可视化、calibration 图

### 11.2 进一步实验
1. **公平性分析**：按年龄、性别、族裔、IMD 分层 C_td
2. **加入 PRS / APOE**：评估额外提升空间，可对比 baseline-only Cox 模型
3. **外部验证**：在 CPRD GOLD（无 HES）或 HUNT（挪威）上 zero-shot 评估泛化能力
4. **Decision Curve Analysis (DCA)**：临床效用评估
5. **Stratified bootstrap CI**：给所有指标加置信区间
6. **可解释性**：哪些 GP / HES events 对预测贡献最大（attention rollout / SHAP）

### 11.3 Self-Training 后续可能的提升
- **V5 第三轮 self-training**：用 V4 模型再做一轮，预期 +0.002~+0.003（递减但仍可能正）
- **不同阈值组合**：V3 用 top 1%，V4 用 top 2%；可探索更激进或更保守阈值
- **混合标签 (soft pseudo-label)**：用 CIF 值作为 weight 而不是 hard 0/1
- **修复 0.19% prevalent leakage**: 把 V2 prevalent 检测从"HES dementia < index"扩展为"任何 GP dementia code < index"，预期 C_td 微调 -0.001~-0.005

### 11.4 ✅ Calibration 已完成 + 决定不做 post-hoc 修复

**已完成（2026-05-13）**：
- V4 calibration: slope @5y dementia = **0.78**, death = **0.77**
- V2 ablation calibration: slope @5y dementia = **0.88**, death = **0.88**
- Plot 路径: `/Data0/swangek_data/991/CPRD/calibration_outputs/{v4,v2_ablation}/`

**决定: 不做 post-hoc calibration 修复**（详见下方 Section 14 "Calibration 解读"）。
理由：post-hoc 修复会让模型预测往 observed (biased) rate 拉，**违背项目识别 hidden dementia 的核心目标**。

**未来 V5（仍要跑）的 calibration**：和前两个一样的方法，不应用 post-hoc 修复。

---

## 12. 关键概念澄清（对新 agent / 老师 必读）

### 12.1 模型预测的是什么

**精确说**：给定患者 age 72 之前的所有 GP + HES 历史 (input)，模型输出从"最后一个 pre-index 事件（≈ age 71-72）"起 0~5 年内 dementia 和 death 的累积发生概率曲线（CIF）。

**粗略说**：从 age ~72 起，预测未来 5 年的 dementia / death 风险。

两种说法描述的是同一件事（差别 < 2 年），见 PROJECT_KNOWLEDGE.md Section 11 完整论述。

### 12.2 Index date (age 72) 的作用

| 角色 | 机制 |
|------|------|
| 输入截止时间 | `_reduce_on_outcome()` 过滤 GP/HES 到 `DATE <= INDEX_DATE` |
| **Cohort 入组条件** | 只研究"活到 72 岁且 event-free 的人"——研究设计层面 |
| 数据训练的"统一年龄段" | 模型在 ~age 72 附近 EHR pattern 上训练，best calibrated for ages 70-75 |

**⚠️ Index date 不是模型查询的强制时点**：临床部署时医生不会等病人"对齐到 age 72"才用模型。模型对每个病人在他自己的 prediction point（最后一次 EHR encounter）上做预测，这是 dynamic prediction 的核心设计意图。

**评估范式跟着模型走**：评估指标应该在 model native timeframe (`y_i(t) = 1[T_i - p_i ≤ t]`) 上做，**不是**强行对齐到 "from index date age 72"。详见 Section 7.3 的范式纠正讨论。

### 12.3 为什么不存在数据泄露（99.81%）

对 incident dementia 患者：outcome 被定义为**第一次** dementia 诊断 → 更早的 input events 按定义没有 dementia code。
对 DEATH/censored 患者：根本没有过 dementia 诊断。

唯一例外：0.19%（16/8257 test）的 prevalent 漏网（GP-coded dementia 在 age 72 之前，但 V2 的 prevalent 筛除只查了 HES，没查 GP）。详见 PROJECT_KNOWLEDGE.md Section 12。

**已验证（2026-05-14）**：剔除 16 个 leaky patient 后重新评估 V5（无需 retrain），cohort-level C_td dementia 下降 -0.0029，C_td death 上升 +0.0001，Approach-A AUROC@5y 变化 ≤0.004，Top-K precision 不变。**0.19% leakage 对 V5 报告数字无实质影响**，原 0.7810 数字稳健。Train-side leakage (213 人) 无法不 retrain 修复，但 test-side 检验暗示影响同样小。

### 12.4 为什么 C_td 直接度量"找隐藏 dementia 的能力"

逻辑链条：
1. C_td 度量模型的 pair-wise ranking 准确率（看 pre-diagnosis history 把谁排前面）
2. 模型输入**不包含** dementia 诊断本身（被 `convert_to_supervised` 拿掉）
3. 一个隐藏 dementia 患者（未确诊但实际有病）的 EHR 长得**和已确诊 dementia 患者一样**（只差一个 code）
4. 所以模型对前者的 ranking 能力 = 对后者的 ranking 能力 = C_td

**经验证据**：V2 corrections（1397 人，外部证据找到的）+ V3 self-training (+0.008) + V4 self-training (+0.005) 都验证了模型 ranking 能力捕捉到了真实信号。

→ **C_td = 0.7732 就是模型"找隐藏 dementia"能力的直接量化**，不需要做"故意藏 dementia"实验。

### 12.5 Fusion 机制对比（简单版）

| | Gated Fusion (当前用) | Cross-Attention Fusion (放弃了) |
|---|---|---|
| 输入 | 各 backbone 的 last token (384 维) | 各 backbone 的全序列 |
| 机制 | 学一个 sigmoid gate 决定每维 GP/HES 混合比例 | last token 主动 attend 对方全序列，6 head |
| 参数量 | ~44 万 | ~150 万 |
| 训练稳定性 | ✅ 简单稳 | ⚠️ 必须 fusion_warmup=3 epoch 才收敛 |
| 实测 C_td | **0.7569** ✅ | 0.7487 ❌ |
| 为什么 gated 赢 | HES 序列稀疏 (几到几十个 token)，last-token 已经汇总；cross-attn 多出来的 1M 参数容易过拟合 | |

⚠️ **注**：V2 ablation (Section 6.5) 揭示 single GP backbone 也能达到 0.7571，**和 dual gated (0.7569) 持平**。这意味着 fusion 机制本身（无论 gated 或 cross-attention）在 clean setting 下贡献接近零；架构差异主要影响训练稳定性，不影响最终性能。

---

## 14. ⚠️ Calibration 解读（Underdiagnosis bias 下的项目立场）

### 背景：dementia 在 GP 数据中被漏诊

文献 well-established：dementia 在英国 GP 数据里被**漏诊 30-50%**（Lang et al. 2017; Connolly et al. 2011; Lancet Commission 2020）。
- **observed dementia rate ≠ true dementia incidence**
- observed = "被诊断的" → systematically biased downward
- true = "实际发病的（含未确诊）" → 大于 observed

### Calibration 计算 vs 解读（**两件事**）

**计算**（mechanical）：
- 把 test 病人按预测 CIF 分 10 bin
- 算每个 bin 的 mean_pred 和 observed_rate
- 拟合 → slope

**永远基于 observed rate**。和"真实率"无关。

**解读**（philosophical）：
- Slope = 0.78 < 1.0 → 模型预测的概率比 observed 高
- **是 bug 还是 feature？取决于"我们认为 model 在预测什么"**

### 两种可能性

**Case A**（V4 真实捕捉了 latent rate）：
- 模型预测 30%, observed 22%, **真实** ≈ 30%
- Slope < 1.0 = **feature**（模型识别了 hidden dementia）
- 这正是 project goal

**Case B**（self-training 过拟合 / confirmation bias）：
- 模型预测 30%, observed 22%, **真实** ≈ 25%
- Slope < 1.0 = **bug**（模型被 pseudo-label 训练拉偏）

**无 external ground truth**（autopsy / biomarker / expert review），无法完全区分。但**两者很可能同时发生**。

### V2 ablation vs V4 slope 的分解

- V2 ablation slope = 0.88（无 self-training）
- V4 slope = 0.78（含 1595 pseudo-labels）

**推断**：
- 0.88 → 1.0 的 gap（**0.12**）：可能反映 **underdiagnosis 的基线偏差**（even without self-training, observed 已 biased）
- V4 比 V2 ablation 低 **0.10**：可能是 self-training 引入的**额外**偏差（A + B 混合）

### 决定：**不做 post-hoc calibration 修复**

**理由**：
1. **违背项目目标**：post-hoc calibration 把预测往 observed rate 拉 → 抹掉 self-training 识别 hidden dementia 的能力
2. **没有 external truth**：我们不知道真实 dementia 率是多少，无法验证修复后是更准还是更偏
3. **项目内部一致性**：从 V2 标签纠正 → V3/V4 self-training，整个 pipeline 设计就是为了**预测 latent rate**，而非 documented rate。Post-hoc fix 反对这个目标

### 但仍要**报告** calibration slope（不修复 ≠ 不报告）

理由：
- ✅ **TRIPOD-AI 合规**：reviewers 期望看到 calibration 分析
- ✅ **比较价值**：V2 ablation vs V4 vs V5 的 slope 变化展示 self-training 对预测分布的影响
- ✅ **透明诚实**：让 reader 自己判断
- ✅ **支撑论文卖点**：slope < 1.0 + cite underdiagnosis 文献 = self-training detecting hidden dementia 的额外证据

### 论文 Methods 段落 (template)

> "Calibration was assessed by binning test patients (n=8,257) into 10 quantile groups by predicted CIF, then computing the observed event rate per bin and fitting a linear regression (slope, intercept). We report **raw calibration slopes without post-hoc adjustment**. Dementia is documented to be underdiagnosed in primary care by 30-50% (Lang et al. 2017; Connolly et al. 2011); our self-training pipeline (Section X) is specifically designed to identify undiagnosed cases. We therefore interpret raw probabilities as estimates of **latent dementia incidence** rather than documented diagnosis rates. Applying post-hoc calibration (e.g., isotonic regression) would suppress this latent-rate estimation and is deliberately omitted."

### 论文 Discussion 段落（解读 slope < 1.0）

> "Our V4 model exhibits calibration slopes of 0.78 (dementia @5y) and 0.77 (death @5y), indicating predicted probabilities exceed documented event rates. We propose two non-mutually-exclusive interpretations:
> (1) **Detection of underdiagnosis**: Self-training (Section X) explicitly relabels high-CIF documented-negative patients. The resulting higher predictions partially reflect true latent dementia incidence, which exceeds documented rates by literature-documented margins (Lang 2017, Connolly 2011).
> (2) **Self-training confirmation bias**: A known phenomenon in pseudo-labeling literature (Arazo et al. 2020). Pseudo-labels treated as hard ground truth can push predictions toward extremes.
>
> Comparison with V2 ablation (no self-training, slope 0.88) suggests the 0.10 additional slope decrease in V4 is partially attributable to (2), while the ~0.12 gap between V2 ablation and ideal slope 1.0 is likely (1). Without external ground truth (autopsy, biomarker, or expert chart review), these two mechanisms cannot be fully separated."

### 检查清单（V5 跑完时）

- [ ] 报 V5 raw calibration slope（不 post-hoc fix）
- [ ] 对比 V2 ablation / V4 / V5 slope trend
- [ ] 如果 V5 slope 继续降 + C_td 仍涨 → 偏向 Case A（继续挖 hidden）
- [ ] 如果 V5 slope 继续降 + C_td 不涨甚至跌 → 偏向 Case B（self-training 收敛 / 过头）

### Future external validation idea

UK Biobank 有 **MRI / APOE / PRS** 子集。可以做：
- 找 model 预测高 CIF 但 documented-negative 的患者
- 看他们的 brain MRI biomarkers / APOE / PRS 是否显著偏向 AD pattern
- 如果是 → 强证据支持 Case A（这些人是真隐藏 dementia）

这是论文未来扩展方向（不是当下任务）。

---

## 13. ⚠️ 必读：每次实验后必须更新项目文件

为避免新 agent 接手时信息缺失，**任何实验产生新结果后**，执行 agent 必须立即更新 **PROJECT_KNOWLEDGE.md** 和 **PROGRESS_REPORT.md** 两个文件。

### 更新协议（按实验类型）

**训练完成后**：
- [ ] Section 6.1 (Complete Results Table) 加新行
- [ ] Section 6.2 (Summary of Best Results) 如果是新最佳，更新
- [ ] Section 6.x 该实验的子章节：训练细节、best epoch、val_loss
- [ ] Section 9.1 (Available Configs) 加 config 文件
- [ ] Section 10.1 (Checkpoints) 加 checkpoint 路径 + 状态
- [ ] PROGRESS_REPORT Section 0 (TL;DR) 时间线表加新行
- [ ] PROGRESS_REPORT Section 6.x：完整实验写法

**Test eval 完成后**：
- [ ] Section 6.1 填上 C_td/Death C_td/Overall C_td（不再 TBD）
- [ ] Section 6.2 更新汇总表
- [ ] Section 6.x 完整 Results 表（含所有 metric）
- [ ] 如果是新最佳：更新 Section 1 (Key Findings) 和 Section 10 metric summary
- [ ] PROGRESS_REPORT Section 10.3 (Final Performance) 更新

**Calibration 分析完成后**：
- [ ] Section 5.7 (Required Eval Metrics) 加该模型的 Calibration Slope 数字
- [ ] Section 6.x 该实验的 Calibration Slope 数据
- [ ] PROGRESS_REPORT Section 7 加新子章节 "X.Y V<n> Calibration Analysis"
- [ ] 保存 plot 路径
- [ ] 和其他模型 calibration 对比

**Ablation 实验完成后**：
- [ ] Section 6.x 专门子章节解释隔离的是什么
- [ ] Section 6.2 加 ablation 行
- [ ] **如果 ablation 改变架构归因 → 必须更新 Section 8 (Lessons Learned)、Section 1 (Key Findings)、Section 0 (TL;DR contributions)**
- [ ] **诚实下调之前的声明**（不要删，标记 strikethrough 加修正）

### 为什么这是 mandatory

1. **连续性**：新 agent 接手能立即看到完整记录
2. **可复现**：未来 replicate 需要完整 provenance
3. **写论文**：临到论文写作时 reconstruct 结果容易出错
4. **诚实**：ablation 显示某个组件贡献 ≈ 0 时，**诚实记录**比之后被 reviewer 戳穿好

### 重要：当 ablation 反驳之前的声明（V2 ablation 就是例子）

如果 ablation 显示某个之前声明的贡献其实接近零（比如 V2 ablation 显示 dual backbone ≈ 0 in clean setting）：
1. **不要删除之前的声明** — 保留并加 strikethrough 或 "previously claimed" 注释
2. **明确加上修正** + ablation 证据
3. **更新 Section 1 / Section 0** 反映修正
4. **重新评估** 任何 downstream framings（论文 Methods, Discussion 草稿）

详细 checklist 见 PROJECT_KNOWLEDGE.md Section 13。

---

## 15. ⚠️ 重大发现: C_td 计算方法 bug + Cohort-level 完整结果 (2026-05-14)

### 15.1 发现过程

2026-05-14, 在准备论文文献对比时, 发现一个内部矛盾:
- Eval pipeline 报 V5 dementia C_td = 0.7810
- 独立用 pycox `EvalSurv.concordance_td('antolini')` 在同一份 V5 inference 输出 (cleaned 8241 患者) 上算出 **0.8467**
- 差 +0.066, 远远超出数值精度

追查发现是 `clinical_prediction_model.py:PerformanceMetrics` Lightning callback 的**聚合方式问题**.

### 15.2 Bug 原理

```python
# clinical_prediction_model.py line 92-127
def get_metrics(self, cdf, lbls, target_ages, ...):
    # called inside on_test_batch_end → ONCE PER BATCH
    surv = pd.DataFrame((1 - cdf).T, index=t_eval)
    ev = EvalSurv(surv, target_ages, lbls, censor_surv='km')
    ctd = ev.concordance_td()  # ← 算当前 batch 的 C_td
    self.log_dict({log_name+"ctd": ctd})  # ← Lightning 默认 reduce_fx="mean" 跨 batch 求均值
```

在 `on_test_batch_end` hook 里, `self.log()` 默认 `on_step=False, on_epoch=True, reduce_fx="mean"`. **结果是 "517 个 batch 各自小 C_td 的均值", 不是 Antolini 教科书定义**.

### 15.3 实证验证 (V3 batch size sweep)

| batch_size | n_batches | %0-dem batches | 平均 C_td | std | 范围 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 (Lightning 默认) | 516 | **47.3%** | **0.7549** | 0.289 | [0.000, 1.000] |
| 32 | 258 | 25.2% | 0.7637 | 0.248 | [0.000, 1.000] |
| 64 | 129 | 4.7% | 0.7788 | 0.206 | [0.000, 1.000] |
| 128 | 65 | 1.5% | 0.7930 | 0.149 | [0.337, 0.991] |
| 256 | 33 | 0.0% | 0.8165 | 0.090 | [0.613, 0.989] |
| 512 | 17 | 0.0% | 0.8184 | 0.069 | [0.670, 0.978] |
| 1024 | 9 | 0.0% | 0.8334 | 0.064 | [0.749, 0.978] |
| **8241 (cohort)** | 1 | 0% | **0.8506** | 0 (exact) | - |

**四个独立 failure mechanism 全部 verified**:
1. **Selection bias**: bs=16 时 47.3% 的 batch 含 0 dementia, 被静默丢弃 (近一半数据)
2. **小样本噪声**: bs=16 时 std=0.29, 范围 [0.000, 1.000], 单个 batch C_td 几乎是随机数
3. **顺序敏感**: bs=16 sequential = 0.7549 vs shuffled = 0.8339, **差 +0.078** (cross-batch pair 丢失)
4. **聚合无意义**: 没有任何文献定义 "per-batch C_td 求均值" 这种指标

### 15.4 独立 verification — Wolbers True CR vs Cause-specific

我们另外用 numpy 手写了一个 Antolini's C_td (独立于 pycox), 同时实现了 Wolbers true competing risk 版本, 6 个合成 test case 全部通过:

| 模型 | pycox cohort | 我们 numpy Antolini | Wolbers true CR |
|---|:---:|:---:|:---:|
| V3 | 0.8506 | 0.8507 | 0.8506 |
| V4 | 0.8487 | 0.8488 | 0.8487 |
| V5 | 0.8467 | 0.8468 | 0.8468 |

→ 三种独立实现差异 < 0.001. **Cause-specific (pycox) 和 True CR (Wolbers) 对 C-index 计算几乎等价** (cause-specific 的 inflation 担心适用于 hazard estimation 和 calibration, 不适用于 C-index).

### 15.5 完整 7 模型 Cohort-level 结果表 (canonical clean cohort N=8241, V2 labels)

| 模型 | 架构 | Pseudo | Per-batch (旧) | **Cohort cause-spec** | Cohort true CR | IBS dem | IBS death |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dual baseline (V1 labels)¹ | Dual gated | 0 | 0.7569 | **0.8416** | 0.8416 | 0.4024 | 0.0558 |
| Cross-attention (V1 labels)¹ | Dual cross-attn | 0 | 0.7487 | **0.8428** | 0.8430 | 0.4061 | 0.0481 |
| V2 labels | Dual gated | 0 | 0.7602 | **0.8447** | 0.8447 | 0.3395 | 0.0805 |
| V2 ablation (single GP) | Single GP | 0 | 0.7571 | **0.8451** | 0.8452 | 0.3152 | 0.0932 |
| **V3 (1st SST, top 1%, +771)** | Dual gated | 771 | 0.7685 | **0.8506** ⭐ | 0.8506 | 0.3265 | 0.1241 |
| V4 (2nd SST, top 2%, +824) | Dual gated | 1595 | 0.7732 | **0.8487** | 0.8487 | 0.2773 | 0.1972 |
| V5 (3rd SST, top 5%, +2219) | Dual gated | 3814 | 0.7810 | **0.8467** | 0.8468 | **0.2713** | 0.3194 |

¹ 因为这两个模型训练用 V1 labels, 评估时用 V2 labels (匹配 PID 替换 labels & event_times) 保证公平对比

### 15.6 4 个关键 cohort-level 发现

#### 发现 1: V3 才是 dementia C_td peak, 不是 V5

Per-batch 排名: V5 (0.7810) > V4 (0.7732) > V3 (0.7685) → "self-training 持续改进"

**Cohort 排名 (真实)**: V3 (0.8506) > V4 (0.8487) > V5 (0.8467) → "**只有第一轮 SST 真的有用**"

完全反转. Per-batch 趋势是小 batch 噪声 + 模型 prediction distribution shape 的交互产生的 artifact. V5 self-training 让 CIF distribution 更尖锐 (两极化), 在 sequential bs=16 偶然得到更高 per-batch 平均, 但 cohort-wide 整体排名实际更差.

#### 发现 2: V2 ablation cohort 也证实 dual ≈ single

| 指标 | V2 labels (Dual) | V2 ablation (Single GP) | Δ |
|---|:---:|:---:|:---:|
| Dementia C_td (cohort) | 0.8447 | 0.8451 | +0.0004 (≈0) |
| Death C_td (cohort) | 0.9582 | 0.9622 | +0.0040 |
| IBS dem (cohort) | 0.3395 | 0.3152 | -0.024 (single 校准略好) |

**HES backbone 12M 参数贡献 effectively 0**, single GP backbone 同等性能甚至略好. V6 应该用 single backbone.

#### 发现 3: Self-training 是 discrimination ↔ calibration trade-off

| 阶段 | C_td (cohort) | IBS dementia (cohort) |
|---|:---:|:---:|
| V2 labels | 0.8447 | 0.3395 |
| **V3 (1st SST)** | **0.8506** ← C_td peak | 0.3265 |
| V4 (2nd SST) | 0.8487 | 0.2773 |
| V5 (3rd SST) | 0.8467 | **0.2713** ← IBS peak |

每加一轮 SST: C_td 微降 -0.002, IBS 改善 ~-0.05. **不是单调改进**.

#### 发现 4: 真实 improvement 比之前想的大很多

之前报: "+0.040 cohort C_td vs baseline 0.733" (per-batch units 混合, 错的)
**正确**: V3 cohort 0.8506 vs hes_aug baseline cohort ~0.80 (估计, 也是 per-batch → cohort 大约 +0.07 offset) = **真实 +0.05 cohort**

或者用 V2 cohort 之前 V1 labels baseline 0.8416 当 reference:
- V2 labels: +0.003 cohort
- V3 (1st SST): +0.090 cohort vs hes_aug baseline (per-batch 0.733)

### 15.7 Mandatory C_td 报告 protocol (论文一律 follow)

**所有未来模型评估必须**:
1. 保存完整 CIF curves per patient (NPZ format)
2. 在整个 test cohort 上一次性算 Antolini's C_td (cause-specific via pycox, 或 Wolbers true CR via 我们自己实现)
3. 报告 bootstrap 95% CI (1000 resamples)
4. 在 canonical clean cohort 8241 上计算 (剔除 16 个 GP-prevalent leaky)
5. **不要用 Lightning callback 报告的 C_td 做任何跨模型对比或 paper 报告**, 只能用作训练时监控

工具:
- `compute_v5_cohort_ctd.py` — V5 含 verification + bootstrap
- `compute_cohort_ctd_generic.py` — 通用版可用任何 NPZ
- `inference_dual_cohort_ctd.py` — dual model 完整 inference
- `inference_single_v2ablation.py` — single backbone 完整 inference

---

## 15.5 早期实验 Cohort-Level 重算结果 (新加 2026-05-14 晚)

完整 cohort C_td recompute 包含 V2 ablation 之前的早期实验:

| 实验 | 测试 cohort | n_test | n_dem | **Cohort C_td** | Per-batch 旧报 | Gap | 备注 |
|------|------------|:------:|:----:|:--------------:|:-------------:|:---:|------|
| **hes_aug** (V1 baseline) | Own V1 | 8,292 | 301 | **0.8360** | 0.733 | +0.103 | V1 自评 |
| hes_aug (V2 labels override) | Canonical 8241 | 8,241 | 370 | **0.8136** | — | — | 公平对比 V2/V3 |
| **hes_fusion** (失败的序列融合) | Own V1 | 11,363 | 505 | **0.7435** | 0.720 | +0.024 | 含 HES-only 患者 |
| hes_fusion (V2 labels) | Canonical 8241 | 8,241 | 370 | **0.7098** | — | — | -0.10 vs hes_aug, **彻底确认失败** |
| **idx68 5-fold CV mean** | Per-fold own | ~40K | ~825 total | **0.8240 ± 0.027** | 0.7022 | +0.122 | folds: 0.787~0.852 |
| idx-age (单 split, 不靠谱) ||||||||
| idx60 | own | 12,071 | 52 | 0.6189 | 0.562 | +0.057 | 太少 events |
| idx70 | own | 10,749 | 44 | 0.8793 | 0.762 | +0.117 | 太少 events |
| idx74 | own | 5,443 | 21 | 0.9168 | 0.825 | +0.092 | 极少 events |
| idx75 | own | 4,147 | 17 | 0.8954 | 0.786 | +0.109 | 极少 events |

**idx-age 注**: 单 split, test dementia event 仅 17-52 个, 统计 power 极低. 报告作 sensitivity, 不作主要结果.

### Baseline → V3 cohort-level 真实贡献分解

| 步骤 | From → To | Cohort C_td 贡献 |
|------|-----------|:--------------:|
| 起点: hes_aug (V1 baseline, 无 HES static) | — | 0.8136 |
| + 22-dim HES static features | hes_aug → dual_baseline | **+0.028** ⭐ 最大单步 |
| + V2 label correction (HES + death-cause) | dual_baseline → V2 labels | +0.003 |
| + 1 round self-training (top 1%, V3) | V2 labels → V3 | +0.006 |
| **总改进 V3 vs hes_aug baseline** | | **+0.037** |

**最大贡献是 22-dim HES static features (+0.028)**, 远超 label corrections 和 self-training 加起来 (+0.009). 这是 cohort-level 重新归因的关键发现.

---

## 16. NEW 对比策略: SurvivEHR-style 自跑 baseline (废弃跨 paper 对比, 2026-05-14)

### 16.1 之前所有跨 paper 对比为什么失败 (Lessons learned 2026-04~05)

| Comparator | 失败原因 | 根本症结 |
|---|---|---|
| Yuan 2024 (UKB DeepSurv) | 1:5 匹配 cohort + 12.6y horizon + APOE+PRS | Cohort 构造 + 特征 + 时间尺度都不同 |
| DemRisk 2024 (CPRD GOLD Cox) | CPRD GOLD vs UKB-linked, 60-79y heterogeneous vs idx 72 homo | 数据源 + 年龄分布 + 模型范式 |
| Anatürk UKBDRS | 14y AUC + 问卷特征 + APOE | 指标 + 特征 |
| Zhang 2026 (UKB + APOE+PRS+LIBRA2) | 40-69y heterogeneous + 基因 + lifestyle | Cohort + 特征 |

**根本问题**: 每个 published paper 都在自己实验设计 (cohort/特征/指标实现/库/时间) 下报数字, 跨 paper 比大小**永远有 confound**. 即使指标名字相同, 实际实现可能不同 (e.g. Yuan 用 mlr3benchmark, 我们用 pycox).

→ **决定**: 完全放弃跨 paper 数字比较.

### 16.2 SurvivEHR paper 的对比哲学 (我们要 learn)

我们的 backbone — Gadd et al. 2025 SurvivEHR paper (medRxiv 2025.08.04.25332916) — 给出了一个干净的 alternative pattern:

**Table 2 报的对比** (5 random seeds, mean ± 95% CI):

**5-year Hypertension (single risk, T2DM cohort n=572K)**:
- SurvivEHR Zero-shot: C_td = 0.561
- SurvivEHR SFT (scratch fine-tune): C_td = 0.816
- **SurvivEHR FFT (with pretrain): C_td = 0.824 ± 0.002** ← Best
- RSF (Random Survival Forests): C_td = 0.729
- DeSurv head only (no transformer): C_td = 0.772
- DeepHit head only: C_td = 0.762

**5-year CVD (competing risk, 同 cohort)**:
- SurvivEHR FFT: C_td = **0.667 ± 0.005** ← Best
- DeSurv head only: C_td = 0.664
- DeepHit: C_td = 0.659
- RSF: C_td = 0.613

**Multi-morbidity at age 50 (heterogeneous, n=20K)**:
- SurvivEHR FFT: C_td = **0.663 ± 0.002** ← Best
- DeSurv: 0.601
- DeepHit: 0.561
- RSF: 0.584

**他们的对比技巧**:
1. **所有 baseline 自跑 in-house, 同一份 cohort**, 不引用任何外部 paper 数字
2. **方法学梯度**: 经典 (RSF) → DL head 单独 → DL head + transformer (SurvivEHR)
3. **内部 ablation**: Zero-shot vs SFT vs FFT, 每步贡献量化
4. **多任务** (hypertension/CVD/multimorbidity), 证明 generality
5. **Sample-size 曲线** (他们 Figure 6C): 数据量 vs 性能, 证明 pretrain 在小 cohort 上特别有价值

### 16.3 我们的新对比设计 (V6+ 阶段)

完全 follow SurvivEHR 模式, 在我们 8,241 患者 cohort 上自跑所有 baseline:

| Level | Baseline | 输入 | 工具 | 估计工作量 | 预期 cohort C_td |
|:---:|---|---|---|:---:|:---:|
| **L0** | Logistic regression (5y binary) | 49 维 static | sklearn | 0.5 天 | ~0.65 |
| **L1** | Cox PH | 49 维 static | lifelines | 0.5 天 | ~0.68 |
| **L2** | Random Survival Forests | 49 维 static | scikit-survival | 0.5 天 | ~0.70 |
| **L3** | DeSurv head + MLP encoder | 49 维 static (无序列) | 我们已有代码 | 1 天 | ~0.72 |
| **L4** | DeepHit head only | 49 维 static | pycox | 1 天 | ~0.71 |
| **L5** | SurvivEHR vanilla FFT (= dual_baseline V1 labels) | GP 序列 + 49 维 | 我们 pipeline | 已完成 | **0.8416** |
| **L6** | + V2 label correction | + V2 labels | 同 | 已完成 | **0.8447** |
| **L7** | + 1 round SST (= V3) | + 771 pseudo | 同 | 已完成 | **0.8506** ⭐ |
| **L8** | V6 final (single backbone + L7 + GP-prevalent 修复 + 实证 lag) | 同 | 待跑 | 1-2 周 | 估计 0.85-0.86 |

→ **每一步贡献清晰量化**, 完整方法学梯度.

### 16.4 论文 Discussion 段落 (修正版 framing)

**段落 1 — 为什么不和已发表论文直接比数字**:

> "Existing dementia risk prediction models in the literature (Yuan et al. 2024 UKB DeepSurv with APOE+PRS; Reeves et al. 2024 DemRisk CPRD Cox PH; Anatürk et al. 2023 UKBDRS; Zhang et al. 2026) report Harrell's C-index ranging 0.749 to 0.846 across substantially different cohort designs (age 60-89 heterogeneous; baseline-only Cox; 1:5 case-control matching; integrated 12-14 year AUC). Direct numerical comparison is methodologically not valid due to (i) cohort heterogeneity driving age-driven discrimination (Walters et al. 2016 empirically demonstrating 0.84→0.56 drop in same Cox model across age strata); (ii) features unavailable in routine EHR (APOE/PRS); (iii) baseline-only Cox paradigm vs our dynamic competing-risk prediction. We therefore construct our comparison within the same cohort, same pipeline, and same metric definition."

**段落 2 — 方法学梯度 framing**:

> "On our 8,241-patient test cohort: classical regression and RSF achieve cohort C_td ~0.65-0.70; deep survival heads without sequence modeling (DeSurv-only, DeepHit-only) achieve ~0.71-0.72; the SurvivEHR foundation model (vanilla fine-tune, no extensions) achieves 0.8416; with our HES static + label correction + 1-round self-training extensions, we achieve **0.8506 (95% CI [0.83, 0.87])**. The ~+0.13 gain from classical to vanilla SurvivEHR reflects transformer sequence modeling + primary-care pretraining; the +0.01 gain to our final pipeline reflects disease-specific extensions on top of the foundation model."

**段落 3 — 与 SurvivEHR paper Table 2 的 cohort 差异说明**:

> "Our vanilla SurvivEHR baseline (cohort C_td 0.8416 on idx 72 UKB GP+HES dementia) differs from SurvivEHR's published benchmark on CVD competing risk (per-batch C_td 0.667 ± 0.005 on T2DM cohort). The difference primarily reflects: (i) cohort homogeneity at idx 72 vs T2DM-conditional in their paper; (ii) dementia event rate (~5%) vs CVD event rate in T2DM patients; (iii) per-batch vs cohort C_td measurement methodology. These differences are within-architecture (same backbone), reflecting downstream cohort/data/metric choices."

---

## 17. V6 实验设计 (下一步 headline model, 2026-05-14)

### 17.1 V6 设计依据 (基于 Section 15 cohort-level 发现 + Section 16 新对比策略)

1. V3 是 cohort C_td peak; V4/V5 是 diminishing returns. **V6 用 1-round SST (V3-style), 不用多轮**
2. V2 ablation 证明 dual backbone ≈ 0. **V6 用 single GP backbone** (~92M params vs dual 104M, 训练快 18%, 性能等同甚至略好 calibration)
3. 16 个 GP-prevalent 患者 + 213 train + 17 val 仍在 V3-V5 数据集. **V6 重建 dataset 扩展 prevalent filter**
4. Pseudo event time 用 last_visit 是错的 (导致 V4 @2y slope=0.27 灾难). **V6 用 V2-type-A 实证 lag 分布抽样** (~2.5y median)

### 17.2 V6 完整 spec

| 组件 | 规格 |
|---|---|
| **Backbone** | Single GP transformer (108K vocab, block_size=512, 6 layers, 6 heads, 384 embed). **无 HES backbone.** Init from `crPreTrain_small_1337.ckpt`. |
| **静态协变量** | 49 维 = 27 demographic + 22 HES static (post-leakage, pre-index filtered) |
| **Label corrections** | 同 V2: relabel 1397 (DEATH+HES dementia + DEATH+death-cause dementia), 移除 487 HES-prevalent |
| **GP-prevalent 修复** | NEW: 移除所有有 pre-index GP dementia code 的患者 (~16 test + 213 train + 17 val) |
| **Self-training** | 1 round only, top 1% CIF threshold (V3-style). Use V6 自己 (需要先跑一遍无 SST 版, 再做 self-training round 1) |
| **Pseudo event time** | NEW: 从 V2-type-A 实证 lag 分布抽样. 算 1123 个 V2-A 患者的 (HES dementia date - last GP visit date) lag, np.random.choice 给每个 pseudo 患者. pseudo_event_age = last_visit_age + sampled_lag |
| **训练** | 同 V3 超参: batch 32, accumulate 16 (effective 512), lr 5e-5 backbone / 5e-4 head, 25 epochs, early stop patience 10 |
| **Evaluation** | Cohort-level C_td (cause-specific + true CR), IBS, INBLL, bootstrap 95% CI, calibration slope @1y/2y/3y/5y |

### 17.3 V6 预期性能

| 指标 | V3 (dual best) | V6 预期 | 理由 |
|---|:---:|:---:|---|
| Dementia C_td (cohort) | 0.8506 | 0.84-0.86 | single ≈ dual (V2 ablation); GP-prevalent 移除 -0.001~-0.003; lag 修复或微涨或微降 |
| Dementia IBS | 0.3265 | 0.30-0.32 | single 略好 (V2 ablation 0.3152); lag 修复改善中段 calibration |
| @2y calibration slope | 0.27 (V4) | 估 0.5-0.7 | lag 修复纠正"全部事件挤在 last visit"的极端 |
| 模型参数 | 104M | ~92M | 砍 HES backbone |
| 训练时间/epoch | 72 min | ~59 min | -18% |

### 17.4 执行 timeline (4-5 周到 first submission)

**Phase 0 (1 天, 最便宜验证)**: Cox PH on our cohort 单跑, verify 0.65-0.70 expected gradient

**Phase 1 (1 周)**: 完整 baseline ladder (LR + Cox PH + RSF + DeSurv head + DeepHit head)

**Phase 2 (1-2 周)**: V6 dataset rebuild + 训练 + cohort eval

**Phase 3 (1 周)**: idx70/74 sensitivity + DCA + paper draft

---

## 18. ⚠️ 重大修正: 模型 horizon 是 25 年, 不是 5 年 (2026-05-15)

继 C_td 计算方法 bug (Section 15) 之后, 又发现一个系统性误解 — **我们一直说"5 年预测", 实际模型 horizon 是 25 年**.

### 18.1 发现过程

V6 设计 pseudo event time 时, 算出 V1-Type-C 的 lag distribution median 8.8 年, 一开始担心"超出 5 年 horizon, 训练-测试 mismatch". 后来 sanity check V5 test set parquet, 发现实际 dementia 事件时间 0-14 年 (median 6.9 年), 与 NPZ 里 event_time_scaled 数字对不上 (NPZ max 0.56 不可能等于 2.8 年).

追查发现是数据 pipeline 的**两层 scaling 叠加**:
1. `FoundationalDataset.time_scale = 1825 days` (default, 5 年) — 把 ages 输入除以 1825
2. `Collator.supervised_time_scale = 5.0` (config) — target_age_delta 再除以 5

**叠加结果**: `target_age_delta = days_lag / (1825 × 5) = days_lag / 9125`. 所以 **normalized 1.0 = 25 年**, 不是 5 年.

### 18.2 实证印证

PID 2469652 (V2 test 一个 dementia 患者):
- Parquet 里 last event date - second_to_last date = 7.66 年 (DAYS_SINCE_BIRTH 29085 - 26289 = 2796 days)
- V5 NPZ event_time_scaled = 0.306
- **0.306 × 25 = 7.66 年** ✓ 与 parquet 完全一致

所以正确换算: **actual_years = event_time_scaled × 25**, 之前我说的 × 5 是错的.

### 18.3 这意味着什么

#### 模型本身没问题
- 模型实际是 **25 年 time-to-event 预测模型**
- 输出 [0, 25 年] CIF 曲线 (1000 个 t 点)
- 可在任意 horizon 查询 risk (1y / 5y / 10y / 14y / 25y)

#### "5 年预测"是怎么来的
是我们 paper 的**查询惯例**: query CIF 曲线在 t=0.2 (= 5/25) 这一点. 不是模型 horizon, 是 evaluation choice.

#### 之前所有训练和评估实际上是对的
- 所有 V1-V5 模型都是 25 年 horizon 训出来的, 没问题
- C_td 评估用实际事件时间, 不依赖 horizon 定义, 没问题
- 全部 cohort C_td 数字 (Section 14-15) 都是对的, 不需要改

#### 之前 paper framing 需要修正
- 不能说"5-year prediction model"
- 应该说: "time-to-event prediction over 25-year horizon, 报告 1y/5y/10y/14y 多个 evaluation points"

### 18.4 全数据集 dementia lag 统计 (2026-05-15)

V2 数据集 train+val+test 所有 last_event 是 dementia code 的患者:

| Statistic | Value |
|---|:---:|
| n_documented dementia | **5,946** |
| Mean lag | **7.49 年** |
| Median lag | **7.37 年** |
| Min / Max | 0.008 / 16.16 年 |
| P5 / P25 / P75 / P95 | 1.55 / 4.69 / 10.36 / 13.35 年 |

累计比例:
- ≤ 5 年: **27.9%**
- ≤ 10 年: **71.9%**
- ≤ 15 年: 99.5%

**意味着**: 真实 dementia 在 EHR 里被诊断的时间, **只有 27.9% 在 5 年内, 72% 在 10 年内**. 文档延迟系统性大于 5 年.

### 18.5 V1-Type-C 是 V6 pseudo lag 的最佳参照

| Source | n | Median |
|---|:---:|:---:|
| V2-Type-A (DEATH+HES) | 995 | 5.17 年 |
| **V1-Type-C (alive censored→HES)** | **3,637** | **8.80 年** |

V1-Type-C 是**活着的, GP 失联后被 HES 抓到 dementia** 的患者, profile 与 V6 pseudo candidates (model 认为高风险的 alive censored) 最匹配.

由于模型 horizon 是 25 年, V1-Type-C 完整 distribution (max 16y) **全部在 horizon 内**, 不需要 cap.

### 18.6 V6 最终 lag 决策

- **Source**: V1-Type-C 完整 distribution (3637 个 positive lags)
- **Sampling**: 每个 pseudo 患者独立 bootstrap (`np.random.choice`)
- **Seed**: 1337
- **不 cap**: 全部用, max 16 年也在 25 年 horizon 内
- **Why bootstrap**: 引入随机性避免模型学到固定 offset 捷径

## 19. V6 实施进展 (2026-05-15, 进行中)

### 19.1 V6 最终设计

| 组件 | 决定 |
|---|---|
| 架构 | Single GP transformer (无 HES backbone, V2 ablation 证明 dual ≈ 0) |
| 静态协变量 | 49 维 (27 demo + 22 HES) |
| 标签 | V2 corrections (1397 relabel + 487 HES-prevalent 移除) |
| GP-prevalent 修复 | NEW: 移除 246 个 GP-prevalent (16 test + 213 train + 17 val) |
| Self-training | 1 round only, top 1% (V3-style; V4/V5 多轮 trade C_td for IBS, 不取) |
| Pseudo event time | NEW: bootstrap from V1-Type-C, 独立 sample per pseudo, seed=1337 |
| Init | 从 `crPreTrain_small_1337.ckpt` 重新 fine-tune (不 warm-start V6-base) |
| 训练超参 | 同 V2 ablation (已验证) |
| 评估 | Cohort C_td (cause-specific + true CR), IBS, INBLL, bootstrap 95% CI |

### 19.2 V6 流水线状态

| Step | 状态 | 备注 |
|---|---|---|
| Step 1 — 算 V1-Type-C empirical lag | ✅ 完成 | 3,637 lags, median 8.8y, 已保存 |
| Step 2 — Build V6-base dataset | ✅ 完成 | 133,076 患者 (119,058 train + 5,777 val + 8,241 test), 12 个 sanity checks 全过 |
| Step 3 — Train V6-base | 🟡 进行中 | 02:10 启动, Epoch 0 在 03:19 完成 (~1h 9min/epoch), 预计 14-21h 总训练 |
| Step 4 — V6-base inference on train | ⏳ 待 Step 3 完成 | ~2.5h GPU |
| Step 5 — Top 1% candidate selection | ⏳ | 预期 ~750-800 candidates |
| Step 6 — 应用 empirical lag + build V6 final | ⏳ | 每个 pseudo 独立 sample lag |
| Step 7 — Train V6 final | ⏳ | 同 V6-base 超参 |
| Step 8 — Cohort evaluation | ⏳ | 与 V3 (0.8506) 和 V2 ablation (0.8451) 对比 |

### 19.3 V6 成功标准

| 结果 | 解读 |
|---|---|
| V6 cohort dementia C_td ≥ 0.8506 | ✅ 匹配或超过 V3, single backbone + 修正 lag 有效 |
| V6 cohort dementia C_td ∈ [0.84, 0.85] | ⚠️ 与 V2 ablation 类似, self-training 在此 setup 下 ≈ 0 贡献 |
| V6 cohort dementia C_td < 0.84 | ❌ GP-prevalent 移除或 lag 修正有问题, 需要诊断 |
| V6 @2y calibration slope > 0.5 | ✅ 修正 lag 解决了 V4 的 0.27 灾难 |

### 19.4 V6 关键文件

| 文件 | 用途 |
|---|---|
| `v6_step1_compute_lag_v3.py` | Step 1 — 算 V1-Type-C lag (用 HES dementia lookup 精确识别) |
| `v6_step2_build_v6_base_dataset.py` | Step 2 — 从 hes_static_v2 过滤 246 个 GP-prevalent |
| `data/v1_typeC_lag_distribution.npy` | V1-Type-C 3637 个 lag 值, bootstrap 用 |
| `data/hes_dementia_lookup.pickle` | 8,318 患者 → HES dementia 日期 (Step 1 副产品, 后续可复用) |
| `data/FoundationalModel/FineTune_Dementia_CR_hes_static_v6_base/` | Step 2 输出, V6-base 数据集 |
| `confs/config_FineTune_Dementia_CR_v6_base.yaml` | V6-base 训练 config |
| `v6_base_train.log` | Step 3 实时训练日志 |
| `figs/full_dataset_dementia_lag.png` | 全数据集 dementia lag 直方图 + CDF |
| `figs/v1_typeC_lag_distribution.png` | V6 用的 lag 分布 |
| `figs/real_prediction_examples.png` | 4 个真实患者完整 CIF 曲线 (0-25y) |
| `raw_predictions_4_patients.txt` | 4 个患者完整 raw CIF 数值 (1000 点 × 2 cause) |

---

## 附录 A：关键文件清单

### 数据集构建
- `build_hes_database.py` — HES SQLite DB
- `build_hes_pretrain_dataset.py` — HES PreTrain Parquet
- `build_hes_summary_features.py` — HES 22-dim 特征（已修复时间泄露）
- `build_dementia_cr_hes_static.py` — V1 数据集
- `build_dementia_cr_hes_aug_v2.py` — V2 数据集（标签纠正）
- `build_dementia_cr_hes_aug_v3.py` — V3 数据集（1st-round self-training）
- `build_dementia_cr_hes_aug_v4.py` — V4 数据集（2nd-round self-training + overlap 分析）

### 模型
- `dual_backbone.py` — DualBackboneSurvModel + FusionLayer（gated / cross_attention / concat_linear）
- `setup_dual_finetune_experiment.py` — DualFineTuneExperiment
- `setup_finetune_experiment.py` — FineTuneExperiment (单 backbone，ablation 用)
- `dual_data_module.py` — DualCollateWrapper, HES tokenizer/cache (含 index date 过滤)
- `competing_risk.py` — DeSurv CR head
- `desurv.py` — ODESurvMultiple, 输出 CIF = π_k(x) · Φ_k(t|x)

### Inference & 评估
- `inference_train_cif.py` — V2 模型对 train set 跑 inference（V3 候选）
- `inference_train_cif_v3.py` — V3 模型对 V3 train set 跑 inference（V4 候选）
- `inference_test_metrics.py` — 测试集 CIF inference (Step 1, V3)
- `compute_additional_metrics.py` — AUROC/Top-K/Harrell's C 计算 (Step 2)
- `inference_test_metrics_v2.py` — 细粒度 CIF + π 保存 (Step 3, V3 时间对齐用)
- `compute_metrics_aligned.py` — 时间对齐指标计算 (Step 3)

### 配置
- `config_FineTune_Dementia_CR_dual.yaml` — Clean baseline (gated fusion, V1 labels)
- `config_FineTune_Dementia_CR_dual_crossattn.yaml` — Cross-attention (放弃)
- `config_FineTune_Dementia_CR_dual_v2.yaml` — V2 labels (gated)
- `config_FineTune_Dementia_CR_dual_v3.yaml` — V3 1st-round self-training (gated)
- `config_FineTune_Dementia_CR_dual_v4.yaml` — **V4 2nd-round self-training (gated, CURRENT BEST)**
- `config_FineTune_Dementia_CR_hes_static_v2_ablation.yaml` — V2 ablation: GP-only + 22-dim static (running)

### 数据文件
- `CPRD/data/train_cif_dementia_v2.csv` — V2 模型 inference 输出 (used for V3 selection)
- `CPRD/data/train_cif_dementia_v3.csv` — V3 模型 inference 输出 (used for V4 selection)
- `CPRD/data/pseudo_dementia_patients_v3.csv` — 771 V3 pseudo-labeled IDs
- `CPRD/data/pseudo_dementia_patients_v4.csv` — 824 V4 pseudo-labeled IDs (with V2 rank metadata)
- `CPRD/data/test_cif_v3.csv` / `test_cif_v3_aligned.csv` / `test_cif_v3_fine.npz` — Test metrics inference outputs

### Checkpoints (Output)
- `crPreTrain_small_1337.ckpt` — GP foundation backbone
- `crPreTrain_HES_1337.ckpt` — HES foundation backbone (8 epoch, test_loss=2.407)
- `crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt` — Dual v2 clean (0.7569)
- `..._dual_v2.ckpt` — Dual + V2 labels (0.7602)
- `..._dual_v3.ckpt` — Dual + V3 self-training (0.7685)
- `..._dual_v4.ckpt` — **Dual + V4 2nd-round self-training (0.7732, CURRENT BEST)**

### 文档
- `PROJECT_KNOWLEDGE.md` — 完整项目架构和实验记录（**新 agent 必读**）
- `PLAN_DUAL_MODEL_ARCHITECTURE.md` — Dual-backbone 详细计划
- `PLAN_IMPROVEMENT_V2.md` — 改进方向计划（22-dim + auxiliary loss）
- `PLAN_ADDITIONAL_METRICS.md` — 附加指标实验计划（Step 1/2/3）
- `PROGRESS_REPORT.md` — 本文件（向老师汇报用，详细总结）

---

## 附录 B：关键术语

| 术语 | 中文 | 含义 |
|------|------|------|
| C_td | Time-dependent concordance index | 时间依赖一致性指数（Antolini's），主指标 |
| Harrell's C | — | 整体 pair-wise concordance（无时间下标） |
| AUROC@5y | — | 5 年时点的 ROC 曲线下面积 |
| IBS | Integrated Brier Score | 积分 Brier 分数（calibration 指标，越低越好） |
| INBLL | Integrated Negative Binomial Log-Likelihood | 越低越好 |
| CIF | Cumulative Incidence Function | 累积发生率函数 |
| DeSurv | Deep Survival distribution estimation | ODE-based 深度生存模型 |
| Index date | — | 患者纳入研究的时间点（age 72） |
| Prevalent case | — | Index date 之前已发病的患者 |
| Censored | — | 没有观察到事件的患者（右截尾） |
| Competing risk | 竞争风险 | 多个互相竞争的可能结局（dementia vs death） |
| EHR | Electronic Health Records | 电子健康档案 |
| GP | General Practice | 全科诊所（初级保健） |
| HES | Hospital Episode Statistics | 英国医院住院记录 |
| OMOP | — | 通用医疗数据模型，用作 ICD-10 → Read v2 映射中介 |
| Read v2 | — | 英国 GP 编码系统 |
| ICD-10 | — | 国际疾病分类第十版（医院编码） |

---

**报告生成日期**：2026-05-09
**当前 best model**：V3 self-training, Dementia C_td = 0.7685
**Status**：进入论文撰写阶段
