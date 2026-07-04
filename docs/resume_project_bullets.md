# 简历项目写法

## 推荐版本

### 中文 AIGC 文本检测与改写鲁棒性评估

- 基于 HC3-Chinese 构建 300 个中文问题的同题对齐数据集，调用 DeepSeek 生成原始回答、口语化改写和对抗式改写，形成 1500 条人类/AI 文本样本。
- 使用 Python 标准库实现 TF-IDF、Logistic Regression、阈值校准、按领域评估和错误分析，完成可复现实验流程。
- 对比 HC3-only、DeepSeek clean aligned、口语改写增强和完整改写增强等训练方案，评估跨生成器和改写攻击场景下的检测鲁棒性。
- 实验发现 HC3-only 模型在 HC3 ChatGPT 上 F1 为 93.88%，但在 DeepSeek 对抗式改写上仅 7.41%；加入同题改写增强后，对抗式改写最佳 F1 提升至 81.40%。

## 更短版本

- 构建 300 个 HC3-Chinese 中文问题的同题 AIGC 检测数据集，使用 DeepSeek 生成原始、口语化和对抗式改写文本，扩展为 1500 条样本。
- 自实现 TF-IDF + Logistic Regression 检测 baseline，完成 question-level split、阈值校准、领域指标、错误分析和可视化输出。
- 发现 HC3-only 检测器在 DeepSeek 对抗式改写上 F1 仅 7.41%，通过改写增强训练最佳提升至 81.40%，验证跨生成器鲁棒性评估的重要性。

## 面试时不要夸大的点

- 不要说这是正式 benchmark，它是科研训练 demo。
- 不要说微调了大模型，目前是轻量 baseline。
- 不要说 DeepSeek 帮你做检测，DeepSeek 只用于生成被检测数据。
- 可以说有真实训练，因为 Logistic Regression 是本地训练出来的。
