# 中文 AIGC 文本伪造检测与改写鲁棒性分析

本项目是一个面向 AIGC 伪造鉴别 / 大模型安全方向的科研训练项目。v2 版本引入公开数据集 HC3-Chinese，并结合自建 DeepSeek 改写攻击集，评估中文 AI 文本检测器在跨数据集、跨生成器和改写攻击下的鲁棒性。

核心问题：

> 一个检测器在公开 HC3-Chinese 的 ChatGPT 文本上表现很好，是否能泛化到 DeepSeek 原始回答、口语化改写和对抗式改写文本？

项目不依赖第三方 Python 包，核心实验使用纯 Python 标准库完成。

## V2 主要升级

- 引入公开数据集 [HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)，使用 6 个中文领域、600 个问题、1200 条人类/ChatGPT 样本
- 保留自建 DeepSeek 改写攻击集：36 个问题，包含原始回答、口语化改写、对抗式改写
- 增加多 baseline：
  - `hc3_char_tfidf_logreg`
  - `hc3_mixed_tfidf_logreg`
  - `hc3_char_tfidf_svm`
  - `aug_char_tfidf_logreg`
  - `aug_mixed_tfidf_logreg`
  - `hc3_stylometric_logreg`
- 增加按领域指标、逐条预测和高置信错误分析
- 实验重点从“做一个检测器”升级为“做检测鲁棒性评估”

## V2 结果摘要

| model | test set | F1 |
| --- | --- | ---: |
| `hc3_char_tfidf_logreg` | HC3 in-domain | 97.18% |
| `hc3_mixed_tfidf_logreg` | HC3 in-domain | 96.59% |
| `hc3_char_tfidf_logreg` | DeepSeek adversarial rewrite | 0.00% |
| `hc3_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 0.00% |
| `aug_mixed_tfidf_logreg` | HC3 in-domain | 97.24% |
| `aug_mixed_tfidf_logreg` | DeepSeek original | 76.92% |
| `aug_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 40.00% |

主要结论：

- 公开 HC3-Chinese 上的同域检测可以达到 96%-97% F1
- 但 HC3-only 模型几乎不能识别 DeepSeek 对抗式改写文本
- 加入 DeepSeek 原始/改写样本训练后，`aug_mixed_tfidf_logreg` 在 DeepSeek 对抗式改写上 F1 提升到 40%
- 这说明跨生成器、跨改写攻击泛化仍然困难，不能只报告干净公开集准确率

## 项目亮点

- 构建 36 个中文问答问题，覆盖 AIGC 检测、大模型安全、医疗 AI、网络工程、后端项目、隐私保护等领域
- 每个问题包含 4 类文本：
  - 人类参考回答
  - DeepSeek 原始回答
  - DeepSeek 口语化改写
  - DeepSeek 对抗式改写
- 实现 `TF-IDF + Logistic Regression` 基础 AI 文本检测器
- 对比两种训练方案：
  - `baseline_original_only`：只用人类参考回答和 DeepSeek 原始回答训练
  - `rewrite_augmented`：加入 DeepSeek 改写样本进行训练增强
- 使用留出验证集进行阈值校准，再在未见过的测试问题上评估
- 输出 Accuracy、Precision、Recall、F1、混淆矩阵、逐条预测、特征权重和实验报告

## 项目结构

```text
aigc_text_detection_demo/
├── data/
│   ├── question_bank.csv
│   ├── deepseek_dataset.csv
│   ├── hc3_chinese_public_subset.csv
│   ├── questions.csv
│   └── sample_dataset.csv
├── docs/
│   ├── github_publish_guide.md
│   ├── interview_guide.md
│   └── project_report.md
├── results/
│   ├── metric_bars.svg
│   ├── metrics.csv
│   ├── metrics_by_domain.csv
│   ├── predictions.csv
│   ├── summary.md
│   └── ...
├── results_v2/
│   ├── v2_metrics.csv
│   ├── v2_metrics_by_domain.csv
│   ├── v2_error_analysis.csv
│   ├── v2_f1_comparison.svg
│   └── v2_summary.md
├── scripts/
│   ├── build_hc3_dataset.py
│   ├── generate_deepseek_dataset.py
│   ├── run_v2_experiment.py
│   └── run_experiment.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 快速运行

进入项目目录：

```powershell
cd "C:\Users\wu060\Documents\New project 3\aigc_text_detection_demo"
```

运行 v2 主实验：

```powershell
python scripts\run_v2_experiment.py
```

运行 v1 DeepSeek demo：

```powershell
python scripts\run_experiment.py
```

核心脚本只使用 Python 标准库，不需要安装 `sklearn`、`pandas` 或 `matplotlib`。

## 重新生成 DeepSeek 数据

如果要重新调用 DeepSeek API：

```powershell
set DEEPSEEK_API_KEY=你的key
python scripts\generate_deepseek_dataset.py
```

脚本会读取：

```text
data/question_bank.csv
```

并生成：

```text
data/deepseek_dataset.csv
results/deepseek_usage.json
```

API key 只从环境变量读取，不会写入任何项目文件。

## V1 实验设计

### 数据划分

- 36 个问题
- 24 个训练问题
- 12 个测试问题

训练问题中再留出 6 个问题做阈值校准：

- fit set：用于训练模型
- calibration set：用于选择最佳分类阈值
- eval set：完全未见过的问题，用于最终评估

### 方法

1. 中文文本归一化，去除空白和常见标点
2. 提取字符级 `1-gram / 2-gram / 3-gram`
3. 计算 TF-IDF 特征
4. 使用 Logistic Regression 二分类
5. 在验证集上选择阈值
6. 在原始文本、口语化改写、对抗式改写三个测试集上评估

## 当前结果

| model config | test set | threshold | accuracy | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline_original_only | original_text | 0.65 | 91.67% | 91.67% | 91.67% | 91.67% |
| baseline_original_only | casual_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% |
| baseline_original_only | adversarial_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% |
| rewrite_augmented | original_text | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% |
| rewrite_augmented | casual_rewrite | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% |
| rewrite_augmented | adversarial_rewrite | 0.86 | 87.50% | 84.62% | 91.67% | 88.00% |

核心观察：

- baseline 对 DeepSeek 原始文本效果较好，F1 为 91.67%
- baseline 对两种改写文本完全失效，F1 为 0.00%
- 加入改写样本训练后，对抗式改写 F1 提升到 88.00%
- 说明 AIGC 文本检测不能只看原始模型输出，还必须评估改写攻击下的鲁棒性

## 主要结果文件

- `results/summary.md`：自动生成的实验总结
- `results/metrics.csv`：总指标
- `results/metrics_by_domain.csv`：按领域统计的指标
- `results/predictions.csv`：逐条样本预测概率
- `results/metric_bars.svg`：F1 对比图
- `results/confusion_matrix_*.svg`：混淆矩阵图
- `results/top_features_baseline_original_only.md`：baseline 特征权重
- `results/top_features_rewrite_augmented.md`：增强模型特征权重
- `results/deepseek_usage.json`：DeepSeek API token 用量记录

## 项目结论

这个 demo 支持一个清晰结论：基础 AI 文本检测器容易学习原始模型输出中的模板化风格特征，但这些特征会被口语化改写和对抗式改写削弱。通过加入改写样本训练和阈值校准，可以显著改善改写场景下的召回率和 F1。

## 局限

- 人类侧是小规模人工整理参考回答，不是大规模独立人类语料
- 数据规模仍然较小，不适合作为正式 benchmark
- 只测试 DeepSeek 单一生成器，尚未覆盖 Qwen、GPT 等多模型场景
- 当前模型是轻量 baseline，后续应加入 Chinese RoBERTa、MacBERT 等更强模型

## 可写进简历

```markdown
### 中文 AIGC 文本伪造检测与改写鲁棒性分析

- 构建 36 个中文问答问题，调用 DeepSeek 生成原始回答、口语化改写和对抗式改写文本，形成 AIGC 文本检测实验数据集
- 使用纯 Python 实现字符级 n-gram、TF-IDF 和 Logistic Regression 检测器，完成训练、阈值校准、评估和可视化分析
- 对比原始文本训练 baseline 与改写增强训练方案，观察 baseline 在改写攻击下 F1 从 91.67% 下降至 0.00%
- 通过加入改写样本训练，将对抗式改写测试集 F1 提升至 88.00%，初步验证改写增强对检测鲁棒性的作用
```

## 参考方向

- M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection  
  https://arxiv.org/abs/2305.14902
- RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors  
  https://arxiv.org/abs/2405.07940
- HC3: Human ChatGPT Comparison Corpus  
  https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
## 构建公开 HC3-Chinese 子集

如果需要重新构建公开数据子集：

```powershell
python scripts\build_hc3_dataset.py
```

该脚本会从 HuggingFace 下载 HC3-Chinese 的 6 个中文领域 JSONL 文件，抽取每个领域 100 个问题，生成：

```text
data/hc3_chinese_public_subset.csv
results_v2/hc3_dataset_summary.json
```

原始 JSONL 文件会放入 `data/raw_hc3_chinese/`，该目录已加入 `.gitignore`。
