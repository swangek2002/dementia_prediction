# 上传到 GitHub 的步骤

## 当前状态

- 远程已添加：`https://github.com/swangek2002/dementia_prediction.git`
- 分支已重命名为 `main`

## 推送命令（需在本地终端执行）

在 Theia 服务器的**终端**中运行（会提示输入 GitHub 用户名和密码/Token）：

```bash
cd /Data0/swangek_data/991
git push -u origin main
```

**注意**：GitHub 已不再支持密码登录，需使用 **Personal Access Token**：
1. GitHub → Settings → Developer settings → Personal access tokens
2. 生成新 token，勾选 `repo` 权限
3. 推送时用 token 代替密码

## 若改用 SSH（推荐，一次配置长期使用）

```bash
# 1. 生成 SSH key（如已有可跳过）
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519 -N ""

# 2. 复制公钥，添加到 GitHub → Settings → SSH and GPG keys
cat ~/.ssh/id_ed25519.pub

# 3. 切换远程为 SSH 并推送
cd /Data0/swangek_data/991
git remote set-url origin git@github.com:swangek2002/dementia_prediction.git
git push -u origin main
```
