# Git 版本管理使用说明

本项目已配置 Git，**只追踪代码**，数据和模型权重已通过 `.gitignore` 排除。

## 常用命令

### 提交前先备份（推荐）
每次让 agent 改代码之前，先提交当前状态：
```bash
cd /Data0/swangek_data/991
git add -A && git commit -m "before agent: 描述即将做的修改"
```

### 查看修改
```bash
git status              # 查看哪些文件被修改
git diff                # 查看具体改动内容
git log --oneline       # 查看提交历史
```

### 回退修改

**丢弃所有未提交的修改，回到上次提交：**
```bash
git checkout -- .
# 或
git restore .
```

**回退到某个历史版本：**
```bash
git log --oneline       # 找到目标 commit 的 hash（如 abc1234）
git checkout abc1234 -- .   # 用该版本覆盖当前工作区
git add -A && git commit -m "revert to abc1234"
```

### 检查是否有大文件被误加
```bash
find . -size +50M -not -path "./.git/*" -not -path "*/.venv/*"
```
如有输出，把对应路径加入 `.gitignore`。

## 已忽略的内容

- 数据：`*.parquet`, `*.db`, `*.csv`, `CPRD/data/` 等
- 模型：`*.ckpt`, `*.pt`, `*.pth`
- 输出：`output/`, `outputs/`, `wandb/`, `checkpoints/`
- 环境：`.venv/`, `conda_envs/`
- 日志：`*.log`, `train_log.txt`

## 注意事项

- **FastEHR** 是嵌套的 git 仓库，当前未纳入本仓库管理
- **CPRD/SurvivEHR** 是指向当前目录的符号链接，已忽略
- 如需追踪 FastEHR，可删除 `FastEHR/.git` 后从 `.gitignore` 中移除 `FastEHR/`
