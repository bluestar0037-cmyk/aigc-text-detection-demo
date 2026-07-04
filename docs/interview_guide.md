# 面试讲解稿：中文 AIGC 文本检测项目

这份文档给你面试前准备用。核心目标是：讲清楚项目动机、数据怎么来、实验怎么做、结果说明什么、局限在哪里、下一步怎么改。

## 1 分钟版本

老师您好，我做的是一个中文 AIGC 文本伪造检测鲁棒性评估项目。最开始我用 36 个 DeepSeek 问答做了一个小 demo，后来发现数据规模和公开性不够，所以升级到了 v2：引入 HC3-Chinese 公开数据集，构建了 600 个问题、1200 条人类/ChatGPT 样本，并保留 DeepSeek 原始回答、口语化改写和对抗式改写作为跨生成器攻击测试集。

方法上，我实现了多个纯 Python baseline，包括字符 TF-IDF + Logistic Regression、mixed TF-IDF + Logistic Regression、Linear SVM 和统计风格特征模型。实验发现，HC3-only 模型在 HC3 同域测试上 F1 可以达到 96% 到 97%，但迁移到 DeepSeek 对抗式改写文本时 F1 会掉到 0。加入 DeepSeek 原始和改写样本训练后，`aug_mixed_tfidf_logreg` 在 DeepSeek 对抗式改写上的 F1 提升到 40%。

这个结果说明，AIGC 文本检测不能只报告公开干净测试集上的准确率，还要关注跨生成器、跨领域和改写攻击下的鲁棒性。

## 3 分钟版本

我的项目题目是“中文 AIGC 文本伪造检测与改写鲁棒性分析”。

项目动机来自一个真实问题：很多 AI 文本检测器在公开数据集或干净 AI 原始输出上表现很好，但真实场景中 AI 生成内容经常会被改写，也可能来自不同生成器。所以我想验证：在 HC3-Chinese 上表现好的检测器，迁移到 DeepSeek 原始回答和改写回答时是否还稳定？

数据方面，我用了两部分。第一部分是公开的 HC3-Chinese，我选取 finance、law、medicine、nlpcc_dbqa、open_qa、psychology 六个领域，每个领域 100 个问题，共 600 个问题、1200 条样本。第二部分是 DeepSeek attack set，包含 36 个问题，每个问题有人类参考回答、DeepSeek 原始回答、口语化改写和对抗式改写。

方法方面，我先做轻量 baseline，没有一上来用深度学习模型。具体包括字符级 n-gram TF-IDF、mixed TF-IDF、Linear SVM 和统计风格特征。这样做的好处是简单、可解释，能作为后续 Chinese RoBERTa 或 MacBERT 的对照。

实验上，我比较了 HC3-only 和 HC3+DeepSeek 增强两类训练方式。HC3-only 模型只学习公开 ChatGPT 风格；增强模型加入 DeepSeek 原始回答和改写回答，观察是否改善跨生成器鲁棒性。

结果上，HC3-only 模型在 HC3 同域测试上 F1 能达到 96%-97%，但对 DeepSeek 对抗式改写 F1 为 0。加入 DeepSeek 改写增强后，`aug_mixed_tfidf_logreg` 在 HC3 同域上仍有 97.24% F1，在 DeepSeek 原始文本上有 76.92% F1，在 DeepSeek 对抗式改写上有 40.00% F1。这个结果不是说我已经解决了检测问题，而是说明跨生成器和改写攻击仍然很难。

我的理解是，这个项目的价值不在于刷最高准确率，而在于把 AIGC 检测做成一个鲁棒性评估流程：公开数据、跨生成器测试、改写攻击、多 baseline、领域指标和错误分析。

## 演示步骤

进入项目：

```powershell
cd "C:\Users\wu060\Documents\New project 3\aigc_text_detection_demo"
```

重新运行实验：

```powershell
python scripts\run_v2_experiment.py
```

展示结果摘要：

```powershell
type results_v2\v2_summary.md
```

展示总指标：

```powershell
type results_v2\v2_metrics.csv
```

可以打开这些图：

- `results_v2/v2_f1_comparison.svg`
- `results_v2/v2_error_analysis.csv`

## 高频问题

### Q1：为什么不用深度学习模型？

可以回答：

> 我先做轻量 baseline，是为了把问题定义、数据构建、评估流程和误差分析跑通。TF-IDF + Logistic Regression 虽然简单，但透明、可解释，适合作为 baseline。后续我会加入 Chinese RoBERTa 或 MacBERT，比较传统方法和预训练模型在改写场景下的鲁棒性。

### Q2：数据是真实 DeepSeek 生成的吗？

可以回答：

> DeepSeek 原始回答、口语化改写和对抗式改写都是真实 API 生成的，生成脚本是 `scripts/generate_deepseek_dataset.py`。v2 的主数据不是我自己编的，而是公开 HC3-Chinese 子集，构建脚本是 `scripts/build_hc3_dataset.py`。

### Q3：为什么 HC3 上效果很高，DeepSeek 改写上会掉到 0？

可以回答：

> 因为 HC3 的 ChatGPT 文本和 DeepSeek 改写文本分布不一样。模型在 HC3 上学到的可能是 ChatGPT 风格、领域分布和表层表达，一旦换成 DeepSeek，并且再做口语化或对抗式改写，原来的判别信号就不稳定了。

### Q4：rewrite_augmented 为什么能改善？

可以回答：

> 因为训练阶段加入了 DeepSeek 原始和改写文本，模型不再只看 HC3 的 ChatGPT 风格，而是接触到另一个生成器和改写变体。不过增强后 DeepSeek adversarial F1 也只有 40%，说明这个问题仍然难，需要更强模型和更多攻击数据。

### Q5：为什么不直接上大模型或 RoBERTa？

可以回答：

> 我先做传统 baseline，是因为 baseline 透明、可复现、能明确暴露鲁棒性问题。后续可以加入 Chinese RoBERTa 或 MacBERT 微调，把它们和当前 TF-IDF/SVM 结果对比，这样才知道深度模型到底提升在哪里。

### Q6：这个项目最大局限是什么？

可以回答：

> 最大局限是数据规模还小，人类侧不是大规模真实人类语料；同时只测试了 DeepSeek 一个生成器。它更像科研入门 demo，不是正式 benchmark。下一步我会扩展公开人类语料、多模型数据和更强检测模型。

### Q7：和乔老师课题组方向怎么对应？

可以回答：

> 课题组方向里有 AIGC 伪造鉴别、大模型安全和大模型测评。我的项目对应的是 AIGC 文本伪造检测，并且关注改写攻击下的鲁棒性，这属于真实应用中很重要的安全问题。

## 你要避免的说法

不要说：

> 我做出了一个很强的 AI 文本检测器。

更稳的说法是：

> 我做了一个完整的小实验，用真实 DeepSeek 数据观察 AIGC 文本检测在改写场景下的鲁棒性问题，并用改写增强训练做了一个初步改善。

## 关键词速记

- AIGC 伪造鉴别：判断内容是否由 AI 生成、篡改或伪造
- 改写攻击：通过口语化、换词、句式重组削弱检测痕迹
- 鲁棒性：输入被扰动后模型性能是否稳定
- TF-IDF：衡量字符 n-gram 在文本中的重要程度
- Logistic Regression：可解释的二分类 baseline
- 阈值校准：在验证集上选择更合适的分类阈值
- Recall：真正 AI 文本中被检测出来的比例
- F1：Precision 和 Recall 的综合指标
- 混淆矩阵：分析 TP、FP、TN、FN

## 最稳自我定位

> 我目前还在 AIGC 安全方向入门阶段，但我已经尝试把一个具体问题拆成公开数据、模型 baseline、跨生成器评估、改写攻击和错误分析几个部分。这个项目让我认识到，AIGC 文本检测不能只看干净公开集准确率，还要评估改写、跨模型和真实应用场景下的鲁棒性。我希望进组后在老师和师兄师姐指导下，把这个方向继续做深。
