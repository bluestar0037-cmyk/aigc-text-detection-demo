# 项目报告：中文 AIGC 文本检测与改写鲁棒性评估

## 1. 项目背景

随着大语言模型生成内容越来越接近人类表达，简单区分“人写文本”和“AI 生成文本”已经不够。真正有价值的问题是：检测器在不同生成器、不同领域、不同改写攻击下是否仍然可靠。

本项目围绕 AIGC 安全中的机器生成文本检测问题，构建一个可复现实验：先用公开 HC3-Chinese 数据训练检测器，再观察它能否泛化到 DeepSeek 原始回答、口语化改写和对抗式改写。项目目标不是做一个线上产品，而是做一个能说明问题的科研 demo。

## 2. 数据构建

v1 项目只有 36 个自建中文问题，能演示“改写会破坏检测器”，但规模偏小。v3 版本把数据升级为同题对齐数据集：

- 从 HC3-Chinese 选取 6 个中文领域：finance、law、medicine、nlpcc_dbqa、open_qa、psychology。
- 每个领域选择 50 个问题，共 300 个问题。
- 每个问题包含 5 类回答：
  - HC3 人类回答
  - HC3 ChatGPT 回答
  - DeepSeek 原始回答
  - DeepSeek 口语化改写
  - DeepSeek 对抗式改写
- 总计 1500 条文本样本。

数据划分采用 question-level split：

| split | questions | expanded samples |
| --- | ---: | ---: |
| train | 210 | 1050 |
| calibration | 42 | 210 |
| eval | 48 | 240 |

这样训练集、阈值校准集和测试集的问题互不重叠，避免模型通过记住同一个问题的措辞获得虚高成绩。

数据质量检查输出在 `results_v3/v3_dataset_quality.json`：

- 当前数据集 300 行，展开后 1500 条文本。
- 缺失字段数量为 0。
- DeepSeek 生成文本长度异常数量为 0。
- 6 个领域各 50 个问题，训练/校准/测试划分完整。

## 3. 方法设计

本项目实现的是一个轻量级、可解释的 baseline，而不是端到端大模型微调。核心流程如下：

1. 文本归一化：转小写、去空白、保留中文、英文和数字字符。
2. 特征提取：使用字符 n-gram 和混合 n-gram 特征。
3. TF-IDF：用训练集统计文档频率，构造稀疏向量。
4. 分类器：使用 Logistic Regression 二分类模型。
5. 阈值校准：在 calibration split 上搜索最佳分类阈值。
6. 鲁棒性评估：在不同生成器和改写攻击场景下分别报告 Accuracy、Precision、Recall、F1。

项目故意使用 Python 标准库实现，目的是让我能在面试中解释每一步，而不是只说“调了一个库”。

## 4. 实验设置

本项目比较 4 类训练策略：

| strategy | training AI variants | purpose |
| --- | --- | --- |
| HC3-only | HC3 ChatGPT | 观察只学公开 ChatGPT 风格是否能泛化 |
| clean aligned | HC3 ChatGPT + DeepSeek original | 观察加入 DeepSeek 原始回答是否足够 |
| casual augmentation | HC3 ChatGPT + DeepSeek original + casual rewrite | 观察普通改写增强能否迁移到更强攻击 |
| full rewrite augmentation | HC3 ChatGPT + DeepSeek original + casual rewrite + adversarial rewrite | 观察完整改写增强的上限 |

测试集始终使用未见过的问题，并分四种条件评估：

- HC3 ChatGPT
- DeepSeek original
- DeepSeek casual rewrite
- DeepSeek adversarial rewrite

每个测试条件都是 48 条人类回答 + 48 条 AI 回答的平衡集合。

## 5. 主要结果

| model | test set | F1 | accuracy |
| --- | --- | ---: | ---: |
| `hc3_mixed_tfidf_logreg` | HC3 ChatGPT | 93.88% | 93.75% |
| `hc3_mixed_tfidf_logreg` | DeepSeek original | 61.33% | 69.79% |
| `hc3_mixed_tfidf_logreg` | DeepSeek casual rewrite | 14.29% | 50.00% |
| `hc3_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 7.41% | 47.92% |
| `clean_aligned_mixed_tfidf_logreg` | DeepSeek original | 85.42% | 85.42% |
| `clean_aligned_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 0.00% | 42.71% |
| `casual_aug_mixed_tfidf_logreg` | DeepSeek casual rewrite | 79.55% | 81.25% |
| `casual_aug_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 62.34% | 69.79% |
| `rewrite_aligned_char_tfidf_logreg` | DeepSeek adversarial rewrite | 81.40% | 83.33% |
| `rewrite_aligned_mixed_tfidf_logreg` | DeepSeek adversarial rewrite | 79.57% | 80.21% |

结论：

- 只在 HC3 ChatGPT 上训练的模型，表面上能达到 93.88% F1，但对 DeepSeek 对抗式改写几乎失效。
- 加入 DeepSeek 原始回答可以提升 DeepSeek original 的检测，但不能解决改写攻击。
- 加入口语化改写后，对抗式改写 F1 提升到 62.34%，说明改写增强具有一定迁移能力。
- 完整加入同题改写增强后，对抗式改写最佳 F1 达到 81.40%，说明同题对齐数据能明显提升检测鲁棒性。

## 6. 错误分析

项目输出 `results_v3/v3_error_analysis.csv`，记录每个模型在每个测试条件下的高置信错误样本。错误样本主要有两类：

- 人类回答被误判为 AI：常见于结构清晰、术语较多、表达正式的回答。
- AI 改写被误判为人类：常见于口语化、短句化、带有不规则表达的 DeepSeek 改写。

这说明轻量检测器容易依赖表层风格特征，而不是理解文本生成来源。因此后续需要更强的语义特征、跨模型数据和更系统的攻击评估。

## 7. 项目价值

相比普通“调用大模型做问答”的 demo，本项目更贴近 AIGC 安全方向：

- 有明确科研问题：检测器是否具备跨生成器和改写攻击鲁棒性。
- 有公开数据基础：HC3-Chinese。
- 有真实模型生成数据：DeepSeek API 生成同题回答和改写。
- 有可复现实验：固定数据划分、固定脚本、固定指标输出。
- 有负结果：HC3-only 对 DeepSeek 改写失效，这比只展示高准确率更有研究价值。
- 有改进实验：通过同题改写增强把对抗式改写最佳 F1 提升到 81.40%。

## 8. 局限性

- DeepSeek 改写由 prompt 生成，还不是严格攻击算法。
- 数据规模仍是 demo 级别，不适合作为正式 benchmark。
- 当前模型是 TF-IDF + Logistic Regression，没有微调 Chinese RoBERTa、MacBERT 等预训练模型。
- 没有覆盖 Qwen、GPT、Claude 等更多生成器。
- 没有做人类标注一致性检查。

## 9. 后续计划

如果继续提升，可以做：

1. 引入 Chinese RoBERTa / MacBERT fine-tuning，与 TF-IDF baseline 对比。
2. 扩大生成器范围：DeepSeek、Qwen、GPT、GLM 等。
3. 加入更多攻击：回译、同义改写、句式扰动、摘要重写。
4. 引入 RAID / M4 的鲁棒性评估思想，构建更系统的 benchmark。
5. 做置信度校准和拒识机制，避免模型在不确定样本上强行判断。

## 10. 面试表述

我会把这个项目定位为一个 AIGC 检测鲁棒性评估 demo。它不是单纯做一个分类器，而是验证一个现象：只在干净公开数据上训练的检测器，遇到新的生成器和改写攻击会明显退化。为了解决这个问题，我构建了同题对齐数据，并用改写增强训练提升了对抗场景下的 F1。
