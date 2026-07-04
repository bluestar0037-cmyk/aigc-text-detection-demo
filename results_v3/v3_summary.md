# V3 Experiment Summary

## What V3 Fixes

- V2 used a large public HC3 subset plus a separate 36-question DeepSeek attack set.
- V3 aligns generators on the same questions: every selected HC3 question has human, ChatGPT, DeepSeek original, casual rewrite, and adversarial rewrite answers.
- The split is question-level, so evaluation questions are unseen during training and threshold calibration.

## Dataset

- Aligned questions: 300
- Domains: 6
- Total text samples if all variants are expanded: 1500
- Train/calibration/eval questions: 210 / 42 / 48
- Eval set size per test condition: 48 human answers + 48 AI answers.

## Training Modes

- `hc3_char_tfidf_logreg`: HC3 only: human vs ChatGPT.
- `hc3_mixed_tfidf_logreg`: HC3 only with mixed char/chunk features.
- `clean_aligned_mixed_tfidf_logreg`: Aligned clean generators, no rewritten DeepSeek text.
- `casual_aug_mixed_tfidf_logreg`: Adds casual rewrites, but not adversarial rewrites.
- `rewrite_aligned_char_tfidf_logreg`: Full aligned rewrite augmentation with char features.
- `rewrite_aligned_mixed_tfidf_logreg`: Full aligned rewrite augmentation with mixed features.

## Main Results

| model | test set | threshold | accuracy | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| casual_aug_mixed_tfidf_logreg | deepseek_adversarial_rewrite | 0.73 | 69.79% | 82.76% | 50.00% | 62.34% |
| casual_aug_mixed_tfidf_logreg | deepseek_casual_rewrite | 0.73 | 81.25% | 87.50% | 72.92% | 79.55% |
| casual_aug_mixed_tfidf_logreg | deepseek_original | 0.73 | 82.29% | 87.80% | 75.00% | 80.90% |
| casual_aug_mixed_tfidf_logreg | hc3_chatgpt | 0.73 | 92.71% | 90.20% | 95.83% | 92.93% |
| clean_aligned_mixed_tfidf_logreg | deepseek_adversarial_rewrite | 0.66 | 42.71% | 0.00% | 0.00% | 0.00% |
| clean_aligned_mixed_tfidf_logreg | deepseek_casual_rewrite | 0.66 | 47.92% | 41.67% | 10.42% | 16.67% |
| clean_aligned_mixed_tfidf_logreg | deepseek_original | 0.66 | 85.42% | 85.42% | 85.42% | 85.42% |
| clean_aligned_mixed_tfidf_logreg | hc3_chatgpt | 0.66 | 91.67% | 87.04% | 97.92% | 92.16% |
| hc3_char_tfidf_logreg | deepseek_adversarial_rewrite | 0.41 | 47.92% | 0.00% | 0.00% | 0.00% |
| hc3_char_tfidf_logreg | deepseek_casual_rewrite | 0.41 | 51.04% | 60.00% | 6.25% | 11.32% |
| hc3_char_tfidf_logreg | deepseek_original | 0.41 | 76.04% | 93.10% | 56.25% | 70.13% |
| hc3_char_tfidf_logreg | hc3_chatgpt | 0.41 | 94.79% | 95.74% | 93.75% | 94.74% |
| hc3_mixed_tfidf_logreg | deepseek_adversarial_rewrite | 0.43 | 47.92% | 33.33% | 4.17% | 7.41% |
| hc3_mixed_tfidf_logreg | deepseek_casual_rewrite | 0.43 | 50.00% | 50.00% | 8.33% | 14.29% |
| hc3_mixed_tfidf_logreg | deepseek_original | 0.43 | 69.79% | 85.19% | 47.92% | 61.33% |
| hc3_mixed_tfidf_logreg | hc3_chatgpt | 0.43 | 93.75% | 92.00% | 95.83% | 93.88% |
| rewrite_aligned_char_tfidf_logreg | deepseek_adversarial_rewrite | 0.82 | 83.33% | 92.11% | 72.92% | 81.40% |
| rewrite_aligned_char_tfidf_logreg | deepseek_casual_rewrite | 0.82 | 85.42% | 92.50% | 77.08% | 84.09% |
| rewrite_aligned_char_tfidf_logreg | deepseek_original | 0.82 | 77.08% | 90.62% | 60.42% | 72.50% |
| rewrite_aligned_char_tfidf_logreg | hc3_chatgpt | 0.82 | 87.50% | 92.86% | 81.25% | 86.67% |
| rewrite_aligned_mixed_tfidf_logreg | deepseek_adversarial_rewrite | 0.78 | 80.21% | 82.22% | 77.08% | 79.57% |
| rewrite_aligned_mixed_tfidf_logreg | deepseek_casual_rewrite | 0.78 | 84.38% | 83.67% | 85.42% | 84.54% |
| rewrite_aligned_mixed_tfidf_logreg | deepseek_original | 0.78 | 77.08% | 80.95% | 70.83% | 75.56% |
| rewrite_aligned_mixed_tfidf_logreg | hc3_chatgpt | 0.78 | 89.58% | 85.19% | 95.83% | 90.20% |

## Key Takeaways

- Best adversarial-rewrite model: `rewrite_aligned_char_tfidf_logreg` with F1=81.40%.
- HC3-only mixed model adversarial F1: 7.41%.
- Full aligned rewrite augmentation adversarial F1: 79.57%.
- Absolute adversarial F1 gain over HC3-only mixed baseline: 72.16%.
- The experiment now measures generator shift and rewrite robustness under a controlled same-question setting.

## Limitations

- DeepSeek rewrites are prompt-generated and should be manually spot-checked before being treated as a formal benchmark.
- Current detectors are lightweight TF-IDF baselines, not fine-tuned transformer detectors.
- The dataset is still a compact research demo, but it is now large and controlled enough to support a credible interview discussion.
