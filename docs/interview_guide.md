# 面试讲解稿：中文 AIGC 文本检测 Demo

这份文档给你面试前准备用。核心目标是：讲清楚项目动机、数据怎么来、实验怎么做、结果说明什么、局限在哪里、下一步怎么改。

## 1 分钟版本

老师您好，我做了一个中文 AIGC 文本伪造检测的小 demo。项目使用 36 个中文问答问题，调用 DeepSeek 生成原始回答、口语化改写回答和对抗式改写回答，再和人类参考回答组成检测数据集。

方法上，我先做了一个透明的 baseline：字符级 n-gram 提取 TF-IDF 特征，再用 Logistic Regression 做二分类，并用验证集校准分类阈值。实验发现，只用原始 DeepSeek 文本训练的 baseline，在原始文本上 F1 有 91.67%，但面对口语化改写和对抗式改写时 F1 下降到 0。后来我加入改写样本做训练增强，对抗式改写测试集 F1 提升到 88%。

这个项目让我理解到，AIGC 文本检测不能只看原始模型输出上的准确率，还要关注改写攻击下的鲁棒性。后续我希望加入更多真实人类语料、多模型生成数据和中文预训练模型，继续深入这个方向。

## 3 分钟版本

我的项目题目是“中文 AIGC 文本伪造检测与改写鲁棒性分析”。

项目动机来自一个真实问题：很多 AI 文本检测器在干净的 AI 原始输出上表现很好，但现实中 AI 生成内容经常会被改写，比如口语化、句式重组、人工润色或另一个模型二次改写。所以我想验证：基础检测器遇到改写后的 AI 文本，会不会明显失效？

数据方面，我设计了 36 个中文问答问题，覆盖 AIGC 检测、大模型安全、医疗 AI、隐私保护、网络工程、后端开发等领域。每个问题有四类文本：人类参考回答、DeepSeek 原始回答、DeepSeek 口语化改写、DeepSeek 对抗式改写。DeepSeek 数据是通过 API 真实生成的，token 用量记录在 `results/deepseek_usage.json`。

方法方面，我先做轻量 baseline，没有一上来用大模型或深度学习。具体是对中文文本做归一化，提取字符级 1 到 3 gram，计算 TF-IDF，再用 Logistic Regression 做二分类。这样做的好处是简单、可解释，也方便观察模型是否依赖“首先、其次、因此”这类表层风格。

实验上，我比较了两种训练方式。第一种是 `baseline_original_only`，只用人类参考回答和 DeepSeek 原始回答训练。第二种是 `rewrite_augmented`，把口语化改写和对抗式改写也加进训练，并做样本平衡。两个模型都用留出的训练问题做阈值校准，然后在完全未见过的 12 个测试问题上评估。

结果很明显：baseline 在原始 DeepSeek 文本上 F1 是 91.67%，但在口语化改写和对抗式改写上 F1 都是 0，说明它主要学到了原始 AI 文本的模板化风格。加入改写训练后，模型在对抗式改写上的 F1 提升到 88%，说明改写增强确实能改善鲁棒性。但增强模型也有误报风险，所以 AIGC 检测需要同时关注召回率和 false positive。

我的理解是，这个项目不是为了说我做出了最强检测器，而是完成了一个完整科研 demo：问题定义、数据构建、baseline、阈值校准、评估、误差分析和局限总结。后续可以加入 HC3 这类公开人类语料，扩展到 Qwen、GPT 等多模型生成文本，再用中文 RoBERTa 或 MacBERT 做更强 baseline。

## 演示步骤

进入项目：

```powershell
cd "C:\Users\wu060\Documents\New project 3\aigc_text_detection_demo"
```

重新运行实验：

```powershell
python scripts\run_experiment.py
```

展示结果摘要：

```powershell
type results\summary.md
```

展示总指标：

```powershell
type results\metrics.csv
```

可以打开这些图：

- `results/metric_bars.svg`
- `results/confusion_matrix_baseline_original_only_original_text.svg`
- `results/confusion_matrix_baseline_original_only_adversarial_rewrite.svg`
- `results/confusion_matrix_rewrite_augmented_adversarial_rewrite.svg`

## 高频问题

### Q1：为什么不用深度学习模型？

可以回答：

> 我先做轻量 baseline，是为了把问题定义、数据构建、评估流程和误差分析跑通。TF-IDF + Logistic Regression 虽然简单，但透明、可解释，适合作为 baseline。后续我会加入 Chinese RoBERTa 或 MacBERT，比较传统方法和预训练模型在改写场景下的鲁棒性。

### Q2：数据是真实 DeepSeek 生成的吗？

可以回答：

> DeepSeek 原始回答、口语化改写和对抗式改写都是真实 API 生成的，生成脚本是 `scripts/generate_deepseek_dataset.py`，结果保存在 `data/deepseek_dataset.csv`，token 用量记录在 `results/deepseek_usage.json`。人类侧目前是小规模人工整理参考回答，后续需要替换或扩展为更大的真实人类语料。

### Q3：为什么 baseline 改写后 F1 会掉到 0？

可以回答：

> baseline 主要依赖原始 AI 文本里的表层风格，比如结构化连接词、句子工整程度和模板化表达。口语化改写会去掉这些痕迹，让文本更像普通学生写的内容，所以模型原来学到的判别信号失效。

### Q4：rewrite_augmented 为什么能改善？

可以回答：

> 因为训练阶段加入了口语化改写和对抗式改写，模型不再只看原始 AI 文本的固定模板，而是接触到更多 AI 生成文本的变体。它对改写后的 AI 文本召回率更高。不过这也可能带来误报，所以我用了验证集做阈值校准。

### Q5：为什么要做阈值校准？

可以回答：

> Logistic Regression 输出的是 AI 概率，如果固定用 0.5，增强模型会比较激进，容易把人类参考文本误判为 AI。我用训练集中留出的 6 个问题做 calibration，选择 F1 较好的阈值。baseline 阈值是 0.65，增强模型阈值是 0.86。

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

> 我目前还在 AIGC 安全方向入门阶段，但我已经尝试把一个具体问题拆成数据、模型、评估和分析四个部分。这个 demo 让我认识到，AIGC 文本检测不能只看干净原始输出上的准确率，还要评估改写、跨模型和真实应用场景下的鲁棒性。我希望进组后在老师和师兄师姐指导下，把这个方向继续做深。
