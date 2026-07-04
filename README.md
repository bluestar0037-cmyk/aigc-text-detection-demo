# 中文 AIGC 文本检测与改写鲁棒性评估

这是一个面向 AIGC 安全、机器生成文本检测和鲁棒性评估的小型科研训练项目。项目从一个 36 题 DeepSeek demo 升级到 v3：基于公开 HC3-Chinese 数据构建 300 个中文问题的同题对齐数据集，并调用 DeepSeek 生成原始回答、口语化改写和对抗式改写，用来观察检测器在跨生成器、跨改写攻击场景下是否可靠。

核心问题：

> 一个检测器如果只在公开 HC3 的 ChatGPT 文本上效果很好，它能否泛化到 DeepSeek 原始回答、口语化改写和对抗式改写？

项目不依赖 `sklearn`、`pandas`、`matplotlib` 等第三方包，核心实验使用 Python 标准库实现，方便复现和解释。

## V3 主要升级

- 数据从“36 个自建问题”升级为“300 个 HC3-Chinese 中文问题”。
- 每个问题保留 5 类文本：HC3 人类回答、HC3 ChatGPT 回答、DeepSeek 原始回答、DeepSeek 口语化改写、DeepSeek 对抗式改写。
- 总样本规模达到 1500 条文本，其中 300 条人类文本、1200 条 AI 文本。
- 采用 question-level split：训练集、阈值校准集、测试集问题互不重叠，避免同题泄漏。
- 比较 HC3-only、DeepSeek clean aligned、casual rewrite augmentation、full rewrite augmentation 等训练方案。
- 输出总指标、按领域指标、逐条预测、错误样本、特征权重和 F1 对比图。

## 数据集

| split | questions | expanded samples |
| --- | ---: | ---: |
| train | 210 | 1050 |
| calibration | 42 | 210 |
| eval | 48 | 240 |

覆盖 6 个中文领域：finance、law、medicine、nlpcc_dbqa、open_qa、psychology。

每个 eval 条件都使用 48 条人类回答 + 48 条对应 AI 回答进行平衡评估。

## V3 关键结果

| model | test set | F1 | accuracy |
| --- | --- | ---: | ---: |
| `hc3_mixed_tfidf_logreg` | HC3 ChatGPT | 93.88% | 93.75% |
| `hc3_mixed_tfidf_logreg` | DeepSeek original | 61.33% | 69.79% |
| `hc3_mixed_tfidf_logreg` | DeepSeek casual rewrite | 14.29% | 50.00% |
| `hc3_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 7.41% | 47.92% |
| `casual_aug_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 62.34% | 69.79% |
| `rewrite_aligned_char_tfidf_logreg` | DeepSeek adversarial rewrite | 81.40% | 83.33% |
| `rewrite_aligned_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 79.57% | 80.21% |

主要观察：

- HC3-only 模型在 HC3 ChatGPT 上 F1 接近 94%，但面对 DeepSeek 改写文本明显失效。
- 只加入 DeepSeek 原始回答可以改善原始 DeepSeek 检测，但不能解决改写攻击。
- 加入口语化改写训练后，对抗式改写 F1 从 7.41% 提升到 62.34%，说明改写增强有迁移效果。
- 加入完整同题改写增强后，对抗式改写最佳 F1 达到 81.40%，证明同题对齐数据能显著提升鲁棒性。

## 项目结构

```text
aigc-text-detection-demo/
├── data/
│   ├── hc3_chinese_public_subset.csv
│   ├── hc3_deepseek_aligned_300.csv
│   └── deepseek_dataset.csv
├── docs/
│   ├── github_publish_guide.md
│   ├── interview_guide.md
│   ├── project_report.md
│   └── resume_project_bullets.md
├── results/
│   └── v1 small demo outputs
├── results_v2/
│   └── public HC3 + separate DeepSeek attack-set outputs
├── results_v3/
│   ├── v3_metrics.csv
│   ├── v3_metrics_by_domain.csv
│   ├── v3_predictions.csv
│   ├── v3_error_analysis.csv
│   ├── v3_f1_comparison.svg
│   ├── v3_summary.md
│   └── deepseek_aligned_usage.json
└── scripts/
    ├── build_hc3_dataset.py
    ├── generate_deepseek_aligned_hc3.py
    ├── run_v3_experiment.py
    ├── run_v2_experiment.py
    └── run_experiment.py
```

## 快速运行

进入项目目录：

```powershell
cd "D:\d盘桌面\aigc-text-detection-demo"
```

运行 v3 主实验：

```powershell
python scripts\run_v3_experiment.py
```

主要输出：

```text
results_v3/v3_metrics.csv
results_v3/v3_metrics_by_domain.csv
results_v3/v3_predictions.csv
results_v3/v3_error_analysis.csv
results_v3/v3_f1_comparison.svg
results_v3/v3_summary.md
```

## 重新生成 DeepSeek 同题数据

如果需要重新调用 DeepSeek API：

```powershell
$env:DEEPSEEK_API_KEY="你的 API key"
python scripts\generate_deepseek_aligned_hc3.py
Remove-Item Env:\DEEPSEEK_API_KEY
```

本项目中的 v3 生成记录：

- model: `deepseek-chat`
- target rows: 300
- 当前数据集行数: 300
- 已记录累计用量：144054 total tokens，不含最初 6 行 dry run
- 最后一次补跑用量见 `results_v3/deepseek_aligned_usage.json`
- 累计用量说明见 `results_v3/deepseek_aligned_usage_total.json`
- API key 只从环境变量读取，不写入任何项目文件

## 方法概述

1. 从 HC3-Chinese 中选取 6 个中文领域，每个领域 50 个问题。
2. 为每个问题保留 HC3 人类回答和 HC3 ChatGPT 回答。
3. 调用 DeepSeek 为同一问题生成三种回答：原始回答、口语化改写、对抗式改写。
4. 使用字符 n-gram / 混合 n-gram 特征构建 TF-IDF 表示。
5. 使用 Logistic Regression 训练二分类检测器。
6. 在 calibration split 上选择分类阈值。
7. 在未见过问题的 eval split 上分别评估 ChatGPT、DeepSeek 原始、DeepSeek 口语化改写、DeepSeek 对抗式改写。

## 可以写进简历

```markdown
### 中文 AIGC 文本检测与改写鲁棒性评估
- 基于 HC3-Chinese 构建 300 个中文问题的同题对齐数据集，调用 DeepSeek 生成原始回答、口语化改写和对抗式改写，形成 1500 条人类/AI 文本样本。
- 使用 Python 标准库实现 TF-IDF、Logistic Regression、阈值校准、按领域评估和错误分析，完成可复现实验流程。
- 对比 HC3-only、DeepSeek clean aligned、改写增强等训练方案，发现 HC3-only 模型在 DeepSeek 对抗式改写上 F1 仅 7.41%，完整改写增强后最佳提升至 81.40%。
- 输出逐条预测、领域指标、错误样本和特征权重分析，验证跨生成器和改写攻击场景下的检测鲁棒性问题。
```

## 参考方向

- HC3: Human ChatGPT Comparison Corpus  
  https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection  
  https://arxiv.org/abs/2305.14902
- RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors  
  https://arxiv.org/abs/2405.07940
