# Experiment Summary

## Setup

- Data source: `deepseek_dataset.csv`
- Questions: 36 total, 24 train, 12 eval
- Text variants: human reference, DeepSeek original, DeepSeek casual rewrite, DeepSeek adversarial rewrite
- Features: Chinese character 1-gram/2-gram/3-gram TF-IDF
- Classifier: pure Python logistic regression

## Training Configs

| model config | fit samples | calibration samples | threshold | vocabulary size | description |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline_original_only | 36 | 12 | 0.65 | 916 | trained on human reference + DeepSeek original answers only |
| rewrite_augmented | 108 | 36 | 0.86 | 3711 | trained on human references + original/casual/adversarial DeepSeek answers with balanced human repeats |

## Results

| model config | test set | threshold | accuracy | precision | recall | F1 | confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_original_only | original_text | 0.65 | 91.67% | 91.67% | 91.67% | 91.67% | TP=11, FP=1, TN=11, FN=1 |
| baseline_original_only | casual_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% | TP=0, FP=1, TN=11, FN=12 |
| baseline_original_only | adversarial_rewrite | 0.65 | 45.83% | 0.00% | 0.00% | 0.00% | TP=0, FP=1, TN=11, FN=12 |
| rewrite_augmented | original_text | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% | TP=10, FP=2, TN=10, FN=2 |
| rewrite_augmented | casual_rewrite | 0.86 | 83.33% | 83.33% | 83.33% | 83.33% | TP=10, FP=2, TN=10, FN=2 |
| rewrite_augmented | adversarial_rewrite | 0.86 | 87.50% | 84.62% | 91.67% | 88.00% | TP=11, FP=2, TN=10, FN=1 |

## Main Observations

- Baseline F1 on original DeepSeek text: 91.67%.
- Baseline F1 on adversarial rewrite: 0.00%.
- Rewrite-augmented F1 on adversarial rewrite: 88.00%.
- The original-only baseline mainly learns obvious style signals from formal DeepSeek outputs.
- Rewritten texts weaken these signals, so robustness must be evaluated separately.
- Adding rewritten samples to training can improve attack-set recall, but may also introduce false positives on human-style writing.

## Limitations

- Human references are small, curated comparison texts rather than a large independent human corpus.
- The dataset is designed for a demo and interview discussion; it is not a formal benchmark.
- A stronger study should add public human corpora, multiple generators, more rewrite strategies, and pretrained Chinese encoders.

## Next Steps

- Replace or expand the human side with HC3-style public human answers or collected student-written answers.
- Add Qwen/GPT-style generators to test cross-model generalization.
- Compare this baseline with Chinese RoBERTa or MacBERT detectors.
- Analyze false positives and false negatives as separate research questions.
