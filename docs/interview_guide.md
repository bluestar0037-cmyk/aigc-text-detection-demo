# 面试讲解稿：AIGC 文本检测与改写鲁棒性评估

## 30 秒版本

老师您好，我做了一个中文 AIGC 文本检测鲁棒性评估项目。它不是简单判断一段话是不是 AI 写的，而是研究检测器在跨生成器和改写攻击下是否还能可靠。我基于 HC3-Chinese 选了 300 个中文问题，并调用 DeepSeek 为同一批问题生成原始回答、口语化改写和对抗式改写，形成 1500 条同题对齐样本。实验发现，只在 HC3 ChatGPT 上训练的模型在 HC3 上 F1 有 93.88%，但在 DeepSeek 对抗式改写上只有 7.41%；加入同题改写增强后，对抗式改写最佳 F1 提升到 81.40%。

## 2 分钟版本

这个项目的动机是：现在很多 AI 文本检测器在干净数据集上表现不错，但用户实际会使用不同模型，还可能做口语化、润色、改写，所以只看 clean accuracy 不够。

我先从公开 HC3-Chinese 数据里选了 6 个中文领域，每个领域 50 个问题，共 300 个问题。每个问题保留 HC3 的人类回答和 ChatGPT 回答，然后用 DeepSeek API 生成三类同题回答：原始回答、口语化改写、对抗式改写。这样每个问题都有 5 种文本，共 1500 条样本。

实验上，我把问题按 train、calibration、eval 划分，保证测试问题在训练中完全没出现。模型部分我没有直接调大模型，而是自己用 Python 标准库实现了 TF-IDF、Logistic Regression、阈值校准、按领域指标和错误分析。这样做的好处是 baseline 可解释，能清楚看到检测器依赖哪些表层风格特征。

结果很明显：HC3-only 模型在 HC3 ChatGPT 上 F1 约 93.88%，但迁移到 DeepSeek adversarial rewrite 时只有 7.41%。加入 DeepSeek 原始回答后，原始 DeepSeek 检测变好，但对改写攻击仍然不够。加入完整改写增强后，对抗式改写最佳 F1 提升到 81.40%。所以我的结论是：AIGC 检测不能只报告干净公开集准确率，必须做跨生成器、跨改写攻击的鲁棒性评估。

## 演示顺序

1. 打开 README，先讲项目目标和 v3 数据升级。
2. 展示 `data/hc3_deepseek_aligned_300.csv`：同一个 question_id 下有 human、ChatGPT、DeepSeek original、casual rewrite、adversarial rewrite。
3. 展示 `scripts/run_v3_experiment.py`：说明自己实现了 TF-IDF、Logistic Regression、阈值校准和评估。
4. 运行：

```powershell
cd "D:\d盘桌面\aigc-text-detection-demo"
python scripts\run_v3_experiment.py
```

5. 展示 `results_v3/v3_metrics.csv` 和 `results_v3/v3_f1_comparison.svg`。
6. 最后讲结论：HC3-only 高分不代表鲁棒，改写增强能显著提升对 DeepSeek 对抗改写的检测。

## 严苛老师可能问什么

### 1. 这个项目是不是只是调 API？

不是。DeepSeek API 只用于生成同题数据，也就是构造被检测对象。检测器本身是我自己实现和训练的，包括文本归一化、n-gram 特征、TF-IDF、Logistic Regression、阈值校准和评估输出。项目重点不是“调用 DeepSeek 回答问题”，而是评估 AI 文本检测器的鲁棒性。

### 2. 有真实训练吗？

有。`run_v3_experiment.py` 会读取 300 个同题问题的数据，把训练集展开成不同训练方案，然后本地训练 Logistic Regression。比如 full rewrite augmentation 模型的训练样本数是 1680，因为每个训练问题会和 4 类 AI 变体配对，并重复人类样本保持类别平衡。

### 3. 为什么不用 DeepSeek 来判断是不是 AI？

因为这个项目研究的是“机器生成文本检测器是否鲁棒”。如果直接让 DeepSeek 判断，实验就变成了调用另一个大模型当裁判，不容易解释，也不一定稳定。我这里先做轻量可解释 baseline，能清楚看到模型在不同生成器和改写攻击下的性能变化。

### 4. 你怎么避免同一个问题泄漏到测试集？

我使用 question-level split。也就是说同一个 question_id 的所有文本变体只会出现在 train、calibration、eval 其中一个集合里，不会出现“训练时见过这个问题的 ChatGPT 回答，测试时再测同题 DeepSeek 改写”的情况。

### 5. 为什么结果里 HC3-only 在 HC3 上很好，但 DeepSeek 改写上很差？

因为 HC3-only 模型主要学习到的是 HC3 ChatGPT 的风格特征，比如更正式、更模板化、更完整的表达。DeepSeek 改写尤其是口语化和对抗式改写，会刻意削弱这些表层特征，所以模型迁移失败。这正是项目想证明的问题：干净数据集成绩不能代表真实鲁棒性。

### 6. 为什么 full rewrite augmentation 后 HC3 ChatGPT 指标反而下降一点？

这是一个合理的 trade-off。加入更多 DeepSeek 改写样本后，模型不再只拟合 HC3 ChatGPT 风格，而是学习更宽的 AI 文本分布，所以在 HC3 单一分布上的 F1 会略降，但在改写攻击上的鲁棒性明显提升。这个现象反而说明实验不是单纯刷分，而是在做跨分布鲁棒性权衡。

### 7. 数据够大吗？

它还不是正式 benchmark，但已经比最初 36 题 demo 可信很多。现在有 300 个同题问题、1500 条文本，且覆盖 6 个中文领域，可以支持一个科研训练项目和面试展示。下一步可以扩展到更多生成器和更多攻击方式。

### 8. 为什么不用 BERT / RoBERTa？

这版先做轻量 baseline，原因是可解释、成本低、方便复现。它的作用是建立清晰的实验框架和下限结果。后续如果进组，我会把同样的数据划分和评估协议迁移到 Chinese RoBERTa、MacBERT 或其他预训练模型上，比较传统特征和深度模型在改写攻击下的差异。

### 9. 你觉得项目最有价值的地方是什么？

最有价值的是实验设计从“做一个分类器”变成了“做鲁棒性评估”。我没有只报告一个高准确率，而是专门测试跨生成器和改写攻击，发现 HC3-only 模型在 DeepSeek adversarial rewrite 上 F1 只有 7.41%，然后用同题改写增强把最佳结果提升到 81.40%。这个过程有问题、实验、负结果、改进和可解释分析。

### 10. 如果继续做，你会怎么做？

我会做四件事：第一，扩大生成器，加入 Qwen、GLM、GPT 等；第二，加入更多攻击，比如回译、摘要重写、句式扰动；第三，微调 Chinese RoBERTa 或 MacBERT；第四，引入更系统的鲁棒性评估协议，比如参考 RAID 和 M4 的多领域、多生成器、多攻击设置。

## 一句话总结

这个项目证明了一个很具体的 AIGC 安全问题：检测器不能只在干净公开数据集上看高分，必须测试跨生成器和改写攻击；同题对齐的改写增强可以显著提升中文 AI 文本检测的鲁棒性。
