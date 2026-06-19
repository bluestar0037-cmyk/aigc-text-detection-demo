# GitHub 上传指南

这份指南用于把 `aigc_text_detection_demo` 上传到 GitHub。

## 方式一：命令行上传

前提：

- 已安装 Git
- 已安装 GitHub CLI
- 已通过 `gh auth login` 登录 GitHub

进入项目目录：

```powershell
cd "D:\d盘桌面\aigc-text-detection-demo"
```

初始化 Git：

```powershell
git init
git branch -M main
```

设置你的用户名和邮箱：

```powershell
git config user.name "bluestar0037-cmyk"
git config user.email "wu060224@outlook.com"
```

检查不要上传 API key：

```powershell
@'
from pathlib import Path
import re
root = Path(".")
secret_re = re.compile("sk-" + r"[A-Za-z0-9]{16,}")
hits = []
for path in root.rglob("*"):
    if path.is_file() and ".git" not in path.parts:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secret_re.search(text):
            hits.append(str(path))
print("SECRET_HITS=" + str(len(hits)))
for h in hits:
    print(h)
'@ | python -
```

如果输出是 `SECRET_HITS=0`，再继续：

```powershell
git add .
git commit -m "Add Chinese AIGC text detection demo"
```

创建 GitHub 仓库并推送：

```powershell
gh repo create aigc-text-detection-demo --public --source=. --remote=origin --push
```

如果你想先设为私有仓库，把 `--public` 改成：

```powershell
--private
```

## 方式二：GitHub Desktop 上传

1. 打开 GitHub Desktop
2. 选择 `File -> Add local repository`
3. 选择目录：

```text
C:\Users\wu060\Documents\New project 3\aigc_text_detection_demo
```

4. 如果提示不是 Git 仓库，选择创建仓库
5. 填写仓库名：

```text
aigc-text-detection-demo
```

6. 点击 `Publish repository`
7. 选择 Public 或 Private
8. 发布前确认没有勾选任何包含 API key 的文件

## 上传前检查清单

- `README.md` 能说明项目做什么、怎么运行、结果是什么
- `data/deepseek_dataset.csv` 不包含 API key
- `.gitignore` 中包含 `.env`
- `results/summary.md` 是最新实验结果
- `python scripts\run_experiment.py` 可以成功运行
- `docs/interview_guide.md` 已经读过一遍，能讲清楚项目

## 仓库描述建议

```text
Chinese AIGC text detection demo with DeepSeek-generated original and rewritten answers, focusing on rewrite robustness.
```

## 仓库标签建议

```text
aigc
llm-security
text-detection
deepseek
robustness
machine-generated-text
```

## 注意

不要把 DeepSeek API key、GitHub token、`.env` 文件上传到仓库。你之前已经在聊天里暴露过一次 DeepSeek key，项目完成后建议去 DeepSeek 控制台删除或重置这个 key。
