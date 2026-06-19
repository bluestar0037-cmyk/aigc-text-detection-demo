"""Run Chinese AIGC text detection experiments.

The experiment uses only the Python standard library:
- character n-gram TF-IDF features
- logistic regression trained with SGD
- baseline vs rewrite-augmented training
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_DATA_PATH = PROJECT_ROOT / "data" / "deepseek_dataset.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


AI_VARIANTS = ["ai_original", "ai_rewrite_casual", "ai_rewrite_adversarial"]
TEST_SETS = {
    "original_text": "ai_original",
    "casual_rewrite": "ai_rewrite_casual",
    "adversarial_rewrite": "ai_rewrite_adversarial",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、,.!?;:（）()《》“”\"'`~\-_/\\[\]{}]", "", text)
    return text


def char_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> list[str]:
    chars = list(normalize(text))
    tokens: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(chars) - n + 1)):
            tokens.append("".join(chars[i : i + n]))
    return tokens


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def records_from_deepseek(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in rows:
        base = {
            "question_id": row["question_id"],
            "split_group": row["split_group"],
            "domain": row["domain"],
            "question": row["question"],
        }
        records.append(
            {
                **base,
                "sample_id": f"{row['question_id']}_human",
                "variant": "human_reference",
                "label": "0",
                "text": row["human_reference"],
            }
        )
        for variant in AI_VARIANTS:
            records.append(
                {
                    **base,
                    "sample_id": f"{row['question_id']}_{variant}",
                    "variant": variant,
                    "label": "1",
                    "text": row[variant],
                }
            )
    return records


def records_from_legacy_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in rows:
        split_group = "train" if row["split"].startswith("train") else "eval"
        records.append(
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "split_group": split_group,
                "domain": row["domain"],
                "question": "",
                "variant": row["variant"],
                "label": row["label"],
                "text": row["text"],
            }
        )
    return records


def load_records() -> tuple[list[dict[str, str]], str]:
    if DEEPSEEK_DATA_PATH.exists():
        return records_from_deepseek(load_csv(DEEPSEEK_DATA_PATH)), "deepseek_dataset.csv"
    return records_from_legacy_sample(load_csv(SAMPLE_DATA_PATH)), "sample_dataset.csv"


def build_vocab(train_rows: list[dict[str, str]], min_df: int = 2) -> tuple[dict[str, int], dict[str, float]]:
    df: Counter[str] = Counter()
    for row in train_rows:
        df.update(set(char_ngrams(row["text"])))

    vocab_tokens = sorted(token for token, count in df.items() if count >= min_df)
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    doc_count = len(train_rows)
    idf = {token: math.log((doc_count + 1) / (df[token] + 1)) + 1.0 for token in vocab}
    return vocab, idf


def vectorize(text: str, vocab: dict[str, int], idf: dict[str, float]) -> dict[int, float]:
    counts: Counter[str] = Counter(char_ngrams(text))
    if not counts:
        return {}

    total = sum(counts.values())
    values: dict[int, float] = {}
    norm_sq = 0.0
    for token, count in counts.items():
        if token not in vocab:
            continue
        value = (count / total) * idf[token]
        idx = vocab[token]
        values[idx] = value
        norm_sq += value * value

    if norm_sq > 0:
        norm = math.sqrt(norm_sq)
        for idx in list(values):
            values[idx] /= norm
    return values


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dot(weights: list[float], features: dict[int, float]) -> float:
    return sum(weights[idx] * value for idx, value in features.items())


def train_logistic_regression(
    rows: list[dict[str, str]],
    vocab: dict[str, int],
    idf: dict[str, float],
    epochs: int = 700,
    learning_rate: float = 0.35,
    l2: float = 0.001,
) -> tuple[list[float], float]:
    vectors = [vectorize(row["text"], vocab, idf) for row in rows]
    labels = [int(row["label"]) for row in rows]
    weights = [0.0 for _ in range(len(vocab))]
    bias = 0.0

    for _ in range(epochs):
        for features, label in zip(vectors, labels):
            pred = sigmoid(dot(weights, features) + bias)
            error = pred - label
            for idx, value in features.items():
                weights[idx] -= learning_rate * (error * value + l2 * weights[idx])
            bias -= learning_rate * error
    return weights, bias


def predict_probability(text: str, vocab, idf, weights, bias) -> float:
    return sigmoid(dot(weights, vectorize(text, vocab, idf)) + bias)


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "total": float(total),
    }


def evaluate(
    rows: list[dict[str, str]],
    vocab,
    idf,
    weights,
    bias,
    threshold: float = 0.5,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    predictions: list[dict[str, str]] = []
    tp = fp = tn = fn = 0

    for row in rows:
        prob = predict_probability(row["text"], vocab, idf, weights, bias)
        pred = 1 if prob >= threshold else 0
        label = int(row["label"])

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "domain": row["domain"],
                "variant": row["variant"],
                "label": str(label),
                "prediction": str(pred),
                "ai_probability": f"{prob:.4f}",
                "text": row["text"],
            }
        )
    return metrics_from_counts(tp, fp, tn, fn), predictions


def split_fit_and_calibration(
    training_rows: list[dict[str, str]],
    calibration_question_count: int = 6,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    question_ids = sorted({row["question_id"] for row in training_rows})
    if len(question_ids) <= calibration_question_count:
        return training_rows, training_rows
    calibration_ids = set(question_ids[-calibration_question_count:])
    fit_rows = [row for row in training_rows if row["question_id"] not in calibration_ids]
    calibration_rows = [row for row in training_rows if row["question_id"] in calibration_ids]
    return fit_rows, calibration_rows


def calibrate_threshold(rows: list[dict[str, str]], vocab, idf, weights, bias) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for i in range(5, 96):
        threshold = i / 100
        metrics, _ = evaluate(rows, vocab, idf, weights, bias, threshold=threshold)
        score = (metrics["f1"], metrics["accuracy"], metrics["recall"], metrics["precision"])
        if best_score is None or score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    assert best_metrics is not None
    return best_threshold, best_metrics


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_training_sets(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    train = [r for r in records if r["split_group"] == "train"]
    baseline = [r for r in train if r["variant"] in {"human_reference", "ai_original", "human", "ai_original"}]

    augmented: list[dict[str, str]] = []
    for r in train:
        if r["variant"] == "human_reference":
            # Balance three AI variants with repeated human references.
            for i in range(3):
                clone = dict(r)
                clone["sample_id"] = f"{clone['sample_id']}_repeat{i}"
                augmented.append(clone)
        elif r["variant"] in set(AI_VARIANTS):
            augmented.append(r)

    return {
        "baseline_original_only": baseline,
        "rewrite_augmented": augmented,
    }


def build_eval_sets(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    eval_records = [r for r in records if r["split_group"] == "eval"]
    eval_sets: dict[str, list[dict[str, str]]] = {}
    for test_name, ai_variant in TEST_SETS.items():
        eval_sets[test_name] = [
            r
            for r in eval_records
            if r["variant"] in {"human_reference", "human", ai_variant}
        ]
    return eval_sets


def evaluate_by_domain(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[row["domain"]].append(row)

    rows: list[dict[str, str]] = []
    for domain, items in sorted(grouped.items()):
        tp = fp = tn = fn = 0
        for item in items:
            pred = int(item["prediction"])
            label = int(item["label"])
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        metrics = metrics_from_counts(tp, fp, tn, fn)
        rows.append(
            {
                "domain": domain,
                "accuracy": f"{metrics['accuracy']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "tp": str(tp),
                "fp": str(fp),
                "tn": str(tn),
                "fn": str(fn),
                "total": str(int(metrics["total"])),
            }
        )
    return rows


def escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_metric_bars(metrics_rows: list[dict[str, str]]) -> None:
    width = 980
    height = 460
    margin_left = 90
    chart_top = 70
    chart_height = 270
    bar_width = 42
    test_names = list(TEST_SETS.keys())
    configs = ["baseline_original_only", "rewrite_augmented"]
    colors = {"baseline_original_only": "#2563eb", "rewrite_augmented": "#16a34a"}
    lookup = {(r["model_config"], r["test_set"]): float(r["f1"]) for r in metrics_rows}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="38" font-size="22" font-family="Arial, Microsoft YaHei" font-weight="700" fill="#111827">F1 under original and rewritten DeepSeek texts</text>',
        f'<line x1="{margin_left}" y1="{chart_top + chart_height}" x2="{width - 40}" y2="{chart_top + chart_height}" stroke="#374151"/>',
        f'<line x1="{margin_left}" y1="{chart_top}" x2="{margin_left}" y2="{chart_top + chart_height}" stroke="#374151"/>',
    ]
    for tick in range(0, 101, 20):
        y = chart_top + chart_height - chart_height * tick / 100
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{tick}%</text>')

    group_width = 250
    for group_idx, test_name in enumerate(test_names):
        group_x = margin_left + 95 + group_idx * group_width
        for config_idx, config in enumerate(configs):
            value = lookup.get((config, test_name), 0.0)
            h = chart_height * value
            x = group_x + config_idx * (bar_width + 18)
            y = chart_top + chart_height - h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{colors[config]}" rx="4"/>')
            parts.append(f'<text x="{x + bar_width / 2}" y="{y - 7:.1f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#111827">{value * 100:.0f}%</text>')
        parts.append(f'<text x="{group_x + 51}" y="{chart_top + chart_height + 32}" text-anchor="middle" font-size="13" font-family="Arial" fill="#111827">{escape_svg(test_name)}</text>')

    parts.extend(
        [
            '<rect x="605" y="26" width="14" height="14" fill="#2563eb" rx="2"/>',
            '<text x="627" y="38" font-size="13" font-family="Arial" fill="#111827">baseline</text>',
            '<rect x="700" y="26" width="14" height="14" fill="#16a34a" rx="2"/>',
            '<text x="722" y="38" font-size="13" font-family="Arial" fill="#111827">rewrite augmented</text>',
            '</svg>',
        ]
    )
    (RESULTS_DIR / "metric_bars.svg").write_text("\n".join(parts), encoding="utf-8")


def write_confusion_matrix_svg(config: str, test_set: str, metrics: dict[str, float]) -> None:
    width = 560
    height = 405
    cell = 112
    x0 = 205
    y0 = 115
    values = [[int(metrics["tn"]), int(metrics["fp"])], [int(metrics["fn"]), int(metrics["tp"])]]
    max_value = max(max(row) for row in values) or 1
    labels = [["TN", "FP"], ["FN", "TP"]]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="19" font-family="Arial, Microsoft YaHei" font-weight="700" fill="#111827">{escape_svg(config)} / {escape_svg(test_set)}</text>',
        f'<text x="{x0 + cell}" y="84" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">Predicted label</text>',
        f'<text x="54" y="{y0 + cell}" transform="rotate(-90 54 {y0 + cell})" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">True label</text>',
        f'<text x="{x0 + cell / 2}" y="{y0 - 15}" text-anchor="middle" font-size="13" font-family="Arial" fill="#374151">Human</text>',
        f'<text x="{x0 + cell * 1.5}" y="{y0 - 15}" text-anchor="middle" font-size="13" font-family="Arial" fill="#374151">AI</text>',
        f'<text x="{x0 - 18}" y="{y0 + cell / 2 + 5}" text-anchor="end" font-size="13" font-family="Arial" fill="#374151">Human</text>',
        f'<text x="{x0 - 18}" y="{y0 + cell * 1.5 + 5}" text-anchor="end" font-size="13" font-family="Arial" fill="#374151">AI</text>',
    ]
    for row in range(2):
        for col in range(2):
            value = values[row][col]
            alpha = 0.18 + 0.72 * value / max_value
            x = x0 + col * cell
            y = y0 + row * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#2563eb" fill-opacity="{alpha:.2f}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + 48}" text-anchor="middle" font-size="17" font-family="Arial" font-weight="700" fill="#111827">{labels[row][col]}</text>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + 78}" text-anchor="middle" font-size="25" font-family="Arial" fill="#111827">{value}</text>')
    parts.append("</svg>")
    filename = f"confusion_matrix_{config}_{test_set}.svg"
    (RESULTS_DIR / filename).write_text("\n".join(parts), encoding="utf-8")


def write_top_features(config: str, vocab: dict[str, int], weights: list[float]) -> None:
    inverse_vocab = {idx: token for token, idx in vocab.items()}
    ranked = sorted(((weight, inverse_vocab[idx]) for idx, weight in enumerate(weights)), reverse=True)
    lines = [
        f"# Top Model Features: {config}",
        "",
        "Positive weights push the classifier toward `AI`; negative weights push it toward `human`.",
        "",
        "## AI-leaning n-grams",
        "",
        "| rank | n-gram | weight |",
        "| --- | --- | ---: |",
    ]
    for i, (weight, token) in enumerate(ranked[:25], 1):
        lines.append(f"| {i} | `{token}` | {weight:.4f} |")
    lines.extend(["", "## Human-leaning n-grams", "", "| rank | n-gram | weight |", "| --- | --- | ---: |"])
    for i, (weight, token) in enumerate(sorted(ranked, key=lambda item: item[0])[:25], 1):
        lines.append(f"| {i} | `{token}` | {weight:.4f} |")
    (RESULTS_DIR / f"top_features_{config}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    data_source: str,
    records: list[dict[str, str]],
    metrics_rows: list[dict[str, str]],
    training_counts: dict[str, int],
    calibration_counts: dict[str, int],
    vocab_sizes: dict[str, int],
    thresholds: dict[str, float],
) -> None:
    lookup = {(r["model_config"], r["test_set"]): r for r in metrics_rows}
    baseline_original = float(lookup[("baseline_original_only", "original_text")]["f1"])
    baseline_adv = float(lookup[("baseline_original_only", "adversarial_rewrite")]["f1"])
    augmented_adv = float(lookup[("rewrite_augmented", "adversarial_rewrite")]["f1"])

    question_count = len({r["question_id"] for r in records})
    train_count = len({r["question_id"] for r in records if r["split_group"] == "train"})
    eval_count = len({r["question_id"] for r in records if r["split_group"] == "eval"})

    lines = [
        "# Experiment Summary",
        "",
        "## Setup",
        "",
        f"- Data source: `{data_source}`",
        f"- Questions: {question_count} total, {train_count} train, {eval_count} eval",
        "- Text variants: human reference, DeepSeek original, DeepSeek casual rewrite, DeepSeek adversarial rewrite",
        "- Features: Chinese character 1-gram/2-gram/3-gram TF-IDF",
        "- Classifier: pure Python logistic regression",
        "",
        "## Training Configs",
        "",
        "| model config | fit samples | calibration samples | threshold | vocabulary size | description |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| baseline_original_only | {training_counts['baseline_original_only']} | {calibration_counts['baseline_original_only']} | {thresholds['baseline_original_only']:.2f} | {vocab_sizes['baseline_original_only']} | trained on human reference + DeepSeek original answers only |",
        f"| rewrite_augmented | {training_counts['rewrite_augmented']} | {calibration_counts['rewrite_augmented']} | {thresholds['rewrite_augmented']:.2f} | {vocab_sizes['rewrite_augmented']} | trained on human references + original/casual/adversarial DeepSeek answers with balanced human repeats |",
        "",
        "## Results",
        "",
        "| model config | test set | threshold | accuracy | precision | recall | F1 | confusion matrix |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['model_config']} | {row['test_set']} | {float(row['threshold']):.2f} | {float(row['accuracy']):.2%} | "
            f"{float(row['precision']):.2%} | {float(row['recall']):.2%} | {float(row['f1']):.2%} | "
            f"TP={row['tp']}, FP={row['fp']}, TN={row['tn']}, FN={row['fn']} |"
        )

    lines.extend(
        [
            "",
            "## Main Observations",
            "",
            f"- Baseline F1 on original DeepSeek text: {baseline_original:.2%}.",
            f"- Baseline F1 on adversarial rewrite: {baseline_adv:.2%}.",
            f"- Rewrite-augmented F1 on adversarial rewrite: {augmented_adv:.2%}.",
            "- The original-only baseline mainly learns obvious style signals from formal DeepSeek outputs.",
            "- Rewritten texts weaken these signals, so robustness must be evaluated separately.",
            "- Adding rewritten samples to training can improve attack-set recall, but may also introduce false positives on human-style writing.",
            "",
            "## Limitations",
            "",
            "- Human references are small, curated comparison texts rather than a large independent human corpus.",
            "- The dataset is designed for a demo and interview discussion; it is not a formal benchmark.",
            "- A stronger study should add public human corpora, multiple generators, more rewrite strategies, and pretrained Chinese encoders.",
            "",
            "## Next Steps",
            "",
            "- Replace or expand the human side with HC3-style public human answers or collected student-written answers.",
            "- Add Qwen/GPT-style generators to test cross-model generalization.",
            "- Compare this baseline with Chinese RoBERTa or MacBERT detectors.",
            "- Analyze false positives and false negatives as separate research questions.",
        ]
    )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records, data_source = load_records()
    training_sets = build_training_sets(records)
    eval_sets = build_eval_sets(records)

    metrics_rows: list[dict[str, str]] = []
    prediction_rows: list[dict[str, str]] = []
    domain_rows: list[dict[str, str]] = []
    training_counts: dict[str, int] = {}
    calibration_counts: dict[str, int] = {}
    vocab_sizes: dict[str, int] = {}
    thresholds: dict[str, float] = {}

    for config_name, train_rows in training_sets.items():
        fit_rows, calibration_rows = split_fit_and_calibration(train_rows)
        training_counts[config_name] = len(fit_rows)
        calibration_counts[config_name] = len(calibration_rows)
        vocab, idf = build_vocab(fit_rows, min_df=2)
        vocab_sizes[config_name] = len(vocab)
        weights, bias = train_logistic_regression(fit_rows, vocab, idf)
        threshold, calibration_metrics = calibrate_threshold(calibration_rows, vocab, idf, weights, bias)
        thresholds[config_name] = threshold
        write_top_features(config_name, vocab, weights)

        for test_name, eval_rows in eval_sets.items():
            metrics, predictions = evaluate(eval_rows, vocab, idf, weights, bias, threshold=threshold)
            metrics_row = {
                "model_config": config_name,
                "test_set": test_name,
                "threshold": f"{threshold:.2f}",
                "calibration_f1": f"{calibration_metrics['f1']:.4f}",
                "calibration_accuracy": f"{calibration_metrics['accuracy']:.4f}",
                "accuracy": f"{metrics['accuracy']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "tp": str(int(metrics["tp"])),
                "fp": str(int(metrics["fp"])),
                "tn": str(int(metrics["tn"])),
                "fn": str(int(metrics["fn"])),
                "total": str(int(metrics["total"])),
            }
            metrics_rows.append(metrics_row)
            write_confusion_matrix_svg(config_name, test_name, metrics)

            for p in predictions:
                prediction_rows.append({"model_config": config_name, "test_set": test_name, **p})
            for d in evaluate_by_domain(predictions):
                domain_rows.append({"model_config": config_name, "test_set": test_name, **d})

    write_csv(
        RESULTS_DIR / "metrics.csv",
        metrics_rows,
        [
            "model_config",
            "test_set",
            "threshold",
            "calibration_f1",
            "calibration_accuracy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "tn",
            "fn",
            "total",
        ],
    )
    write_csv(
        RESULTS_DIR / "predictions.csv",
        prediction_rows,
        ["model_config", "test_set", "sample_id", "question_id", "domain", "variant", "label", "prediction", "ai_probability", "text"],
    )
    write_csv(
        RESULTS_DIR / "metrics_by_domain.csv",
        domain_rows,
        ["model_config", "test_set", "domain", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "total"],
    )
    write_csv(
        RESULTS_DIR / "dataset_overview.csv",
        [
            {
                "data_source": data_source,
                "questions_total": str(len({r["question_id"] for r in records})),
                "train_questions": str(len({r["question_id"] for r in records if r["split_group"] == "train"})),
                "eval_questions": str(len({r["question_id"] for r in records if r["split_group"] == "eval"})),
                "records_total": str(len(records)),
            }
        ],
        ["data_source", "questions_total", "train_questions", "eval_questions", "records_total"],
    )
    write_metric_bars(metrics_rows)
    write_summary(data_source, records, metrics_rows, training_counts, calibration_counts, vocab_sizes, thresholds)

    print("Experiment finished.")
    print(f"Data source: {data_source}")
    print(f"Results directory: {RESULTS_DIR}")
    for row in metrics_rows:
        print(
            f"{row['model_config']} / {row['test_set']}: "
            f"threshold={float(row['threshold']):.2f}, accuracy={float(row['accuracy']):.2%}, "
            f"recall={float(row['recall']):.2%}, f1={float(row['f1']):.2%}"
        )


if __name__ == "__main__":
    main()
