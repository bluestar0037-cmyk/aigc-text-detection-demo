# 项目报告：中文 AIGC 文本伪造检测与改写鲁棒性分析

## 1. 背景

随着大语言模型生成能力增强，AI 生成文本在问答、作业、评论、新闻和知识科普等场景中越来越常见。判断一段文本是否由 AI 生成，是 AIGC 伪造鉴别中的基础问题。

但真实场景中的 AI 文本往往不会保持原始输出形态。它可能经过口语化改写、句式重组、人工润色或另一个模型的二次改写。因此，一个检测器如果只在原始 AI 输出上表现好，并不能说明它在真实场景中可靠。

本项目围绕“改写是否会削弱 AI 文本检测器”设计了一个小规模实验。

## 2. 实验目标

本项目关注三个问题：

1. 只用 DeepSeek 原始回答训练的基础检测器，能否识别未见过问题上的原始 DeepSeek 文本？
2. 同一个检测器面对口语化改写和对抗式改写时是否会失效？
3. 如果把改写样本加入训练，能否提升检测器在改写场景下的鲁棒性？

## 3. 数据构建

项目构建了 36 个中文问答问题，覆盖以下领域：

- AIGC 文本检测
- 大模型安全测评
- 医疗 AI
- 数据隐私
- 网络工程
- 后端项目开发
- 编程学习
- 科研训练

每个问题包含四类文本：

- 人类参考回答
- DeepSeek 原始回答
- DeepSeek 口语化改写回答
- DeepSeek 对抗式改写回答

DeepSeek 数据由 `scripts/generate_deepseek_dataset.py` 调用 API 生成，并保存到 `data/deepseek_dataset.csv`。

## 4. 方法

本项目使用一个可解释的轻量 baseline：

- 特征：中文字符级 1-gram、2-gram、3-gram
- 表示：TF-IDF
- 分类器：Logistic Regression
- 实现：纯 Python 标准库

选择该 baseline 的原因：

- 对本科科研 demo 足够透明
- 不依赖 GPU 和大型机器学习库
- 可以通过特征权重观察模型学到的表层语言特征
- 适合作为后续 RoBERTa / MacBERT 等模型的对照组

## 5. 训练与评估

数据划分：

- 24 个训练问题
- 12 个测试问题

训练问题中再留出 6 个问题用于阈值校准。最终在未见过的 12 个测试问题上评估。

实验比较两种训练方案：

| model config | 说明 |
| --- | --- |
| baseline_original_only | 只用人类参考回答和 DeepSeek 原始回答训练 |
| rewrite_augmented | 加入 DeepSeek 口语化改写和对抗式改写，并对人类参考样本做平衡重复 |

评估场景：

- original_text：人类参考回答 vs DeepSeek 原始回答
- casual_rewrite：人类参考回答 vs DeepSeek 口语化改写
- adversarial_rewrite：人类参考回答 vs DeepSeek 对抗式改写

## 6. 实验结果

| model config | test set | threshold | accuracy | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline_original_only | original_text | 0.65 | 91.67% | 91.67% | 91.67% | 91.67% |
| baseline_original_only | casual_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% |
| baseline_original_only | adversarial_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% |
| rewrite_augmented | original_text | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% |
| rewrite_augmented | casual_rewrite | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% |
| rewrite_augmented | adversarial_rewrite | 0.86 | 87.50% | 84.62% | 91.67% | 88.00% |

## 7. 分析

baseline 在原始 DeepSeek 文本上表现较好，说明 DeepSeek 原始回答中存在比较明显的结构化和模板化风格，例如“首先、其次、此外、因此、建议”等表达。

但面对口语化改写和对抗式改写时，baseline 的召回率降为 0。这说明模型学到的主要是表层风格特征，而不是稳定的“AI 来源”特征。

rewrite_augmented 在训练阶段加入改写样本后，改写测试集 F1 提升到 83% 到 88%。这说明改写增强能提升鲁棒性，但也带来新的问题：模型可能更容易把自然、工整的人类表达误判为 AI。因此 AIGC 检测需要同时关注召回率和误报率。

## 8. 局限

- 人类参考回答规模较小，不能代表真实人类写作分布
- 数据集是 demo 级别，不是正式 benchmark
- 当前只使用 DeepSeek 单一生成器，没有测试跨模型泛化
- 改写策略只有两类，真实攻击方式会更多样
- baseline 较简单，后续需要与中文预训练模型对比

## 9. 后续计划

- 引入 HC3 等公开人类语料，扩大人类侧样本
- 加入 Qwen、GPT 等不同生成模型，测试跨模型泛化
- 增加同义替换、摘要压缩、学术化改写、人工润色等攻击方式
- 使用 Chinese RoBERTa、MacBERT 等模型作为强 baseline
- 分析 false positive 和 false negative，研究误判背后的语言特征

## 10. 总结

本项目通过真实 DeepSeek 数据验证了 AIGC 文本检测中的改写鲁棒性问题：基础检测器在原始模型输出上可以取得较好效果，但面对改写文本时可能完全失效。加入改写样本训练和阈值校准后，模型在改写场景下的 F1 明显提升。这说明 AIGC 伪造检测不能只看干净数据上的准确率，还需要系统评估对抗改写和真实应用场景下的稳定性。
