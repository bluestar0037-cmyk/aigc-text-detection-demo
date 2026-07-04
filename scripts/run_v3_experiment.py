"""Run the v3 aligned HC3 + DeepSeek robustness experiment.

V3 fixes the main weakness of V2: the public HC3 data and the custom
DeepSeek attack data are now aligned by question. Each selected question has:

- HC3 human answer
- HC3 ChatGPT answer
- DeepSeek original answer
- DeepSeek casual rewrite
- DeepSeek adversarial rewrite

The question-level split is preserved, so train/calibration/eval questions do
not overlap. The script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hc3_deepseek_aligned_300.csv"
RESULTS_DIR = PROJECT_ROOT / "results_v3"

TEXT_COLUMNS = {
    "human_hc3": ("human_hc3", 0),
    "chatgpt_hc3": ("chatgpt_hc3", 1),
    "deepseek_original": ("deepseek_original", 1),
    "deepseek_casual_rewrite": ("deepseek_rewrite_casual", 1),
    "deepseek_adversarial_rewrite": ("deepseek_rewrite_adversarial", 1),
}

TEST_SETS = {
    "hc3_chatgpt": ["chatgpt_hc3"],
    "deepseek_original": ["deepseek_original"],
    "deepseek_casual_rewrite": ["deepseek_casual_rewrite"],
    "deepseek_adversarial_rewrite": ["deepseek_adversarial_rewrite"],
}

MODEL_SPECS = [
    {
        "model": "hc3_char_tfidf_logreg",
        "feature_kind": "tfidf_char",
        "learner": "logreg",
        "train_ai_variants": ["chatgpt_hc3"],
        "note": "HC3 only: human vs ChatGPT.",
    },
    {
        "model": "hc3_mixed_tfidf_logreg",
        "feature_kind": "tfidf_mixed",
        "learner": "logreg",
        "train_ai_variants": ["chatgpt_hc3"],
        "note": "HC3 only with mixed char/chunk features.",
    },
    {
        "model": "clean_aligned_mixed_tfidf_logreg",
        "feature_kind": "tfidf_mixed",
        "learner": "logreg",
        "train_ai_variants": ["chatgpt_hc3", "deepseek_original"],
        "note": "Aligned clean generators, no rewritten DeepSeek text.",
    },
    {
        "model": "casual_aug_mixed_tfidf_logreg",
        "feature_kind": "tfidf_mixed",
        "learner": "logreg",
        "train_ai_variants": ["chatgpt_hc3", "deepseek_original", "deepseek_casual_rewrite"],
        "note": "Adds casual rewrites, but not adversarial rewrites.",
    },
    {
        "model": "rewrite_aligned_char_tfidf_logreg",
        "feature_kind": "tfidf_char",
        "learner": "logreg",
        "train_ai_variants": [
            "chatgpt_hc3",
            "deepseek_original",
            "deepseek_casual_rewrite",
            "deepseek_adversarial_rewrite",
        ],
        "note": "Full aligned rewrite augmentation with char features.",
    },
    {
        "model": "rewrite_aligned_mixed_tfidf_logreg",
        "feature_kind": "tfidf_mixed",
        "learner": "logreg",
        "train_ai_variants": [
            "chatgpt_hc3",
            "deepseek_original",
            "deepseek_casual_rewrite",
            "deepseek_adversarial_rewrite",
        ],
        "note": "Full aligned rewrite augmentation with mixed features.",
    },
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\u4e00-\u9fff0-9a-z]+", "", text)


def char_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> list[str]:
    chars = list(normalize(text))
    tokens: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(chars) - n + 1)):
            tokens.append("".join(chars[i : i + n]))
    return tokens


def mixed_tokens(text: str) -> list[str]:
    clean = normalize(text)
    tokens = char_ngrams(text, 2, 4)
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", clean)
    tokens.extend(f"W:{chunk}" for chunk in chunks)
    return tokens


def make_sample(row: dict[str, str], variant: str, pair_variant: str) -> dict[str, str]:
    column, label = TEXT_COLUMNS[variant]
    return {
        "sample_id": f"{row['question_id']}_{variant}__pair_{pair_variant}",
        "question_id": row["question_id"],
        "dataset": "HC3-DeepSeek-Aligned",
        "domain": row["domain"],
        "split": row["split"],
        "variant": variant,
        "pair_variant": pair_variant,
        "label": str(label),
        "question": row["question"],
        "text": row[column],
    }


def build_pair_records(rows: list[dict[str, str]], ai_variants: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in rows:
        for variant in ai_variants:
            records.append(make_sample(row, "human_hc3", variant))
            records.append(make_sample(row, variant, variant))
    return records


def build_vocab(rows: list[dict[str, str]], tokenizer, min_df: int = 2) -> tuple[dict[str, int], dict[str, float]]:
    df: Counter[str] = Counter()
    for row in rows:
        df.update(set(tokenizer(row["text"])))
    vocab_tokens = sorted(token for token, count in df.items() if count >= min_df)
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    doc_count = len(rows)
    idf = {token: math.log((doc_count + 1) / (df[token] + 1)) + 1.0 for token in vocab}
    return vocab, idf


def vectorize_tfidf(text: str, tokenizer, vocab: dict[str, int], idf: dict[str, float]) -> dict[int, float]:
    counts: Counter[str] = Counter(tokenizer(text))
    total = sum(counts.values()) or 1
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


def dot(weights: list[float], features: dict[int, float]) -> float:
    return sum(weights[idx] * value for idx, value in features.items())


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def train_logreg(
    vectors: list[dict[int, float]],
    labels: list[int],
    dim: int,
    *,
    epochs: int = 16,
    lr: float = 0.55,
    l2: float = 0.0007,
    seed: int = 2026,
) -> tuple[list[float], float]:
    weights = [0.0] * dim
    bias = 0.0
    rng = random.Random(seed)
    order = list(range(len(labels)))
    for epoch in range(epochs):
        rng.shuffle(order)
        current_lr = lr / math.sqrt(epoch + 1)
        for idx in order:
            feats = vectors[idx]
            label = labels[idx]
            pred = sigmoid(dot(weights, feats) + bias)
            error = pred - label
            for feat_idx, value in feats.items():
                weights[feat_idx] -= current_lr * (error * value + l2 * weights[feat_idx])
            bias -= current_lr * error
    return weights, bias


def counts_to_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def evaluate(rows, vectorizer, weights, bias, threshold: float) -> tuple[dict[str, float | int], list[dict[str, str]]]:
    predictions = []
    tp = fp = tn = fn = 0
    for row in rows:
        score = sigmoid(dot(weights, vectorizer(row["text"])) + bias)
        pred = 1 if score >= threshold else 0
        label = int(row["label"])
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
        predictions.append({**row, "prediction": str(pred), "ai_probability": f"{score:.4f}"})
    return counts_to_metrics(tp, fp, tn, fn), predictions


def calibrate(rows, vectorizer, weights, bias) -> tuple[float, dict[str, float | int]]:
    best = None
    for i in range(5, 96):
        threshold = i / 100
        metrics, _ = evaluate(rows, vectorizer, weights, bias, threshold)
        score = (
            float(metrics["f1"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def format_metric_row(row: dict[str, str | float | int]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.4f}"
        else:
            out[key] = str(value)
    return out


def domain_metrics(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in predictions:
        groups[row["domain"]].append(row)
    out = []
    for domain, rows in sorted(groups.items()):
        tp = fp = tn = fn = 0
        for row in rows:
            pred, label = int(row["prediction"]), int(row["label"])
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        out.append(format_metric_row({"domain": domain, **counts_to_metrics(tp, fp, tn, fn)}))
    return out


def top_errors(predictions: list[dict[str, str]], model_name: str, test_set: str, limit: int = 20) -> list[dict[str, str]]:
    errors = [row for row in predictions if row["label"] != row["prediction"]]
    errors.sort(key=lambda row: abs(float(row["ai_probability"]) - 0.5), reverse=True)
    return [
        {
            "model": model_name,
            "test_set": test_set,
            "sample_id": row["sample_id"],
            "question_id": row["question_id"],
            "domain": row["domain"],
            "variant": row["variant"],
            "pair_variant": row["pair_variant"],
            "label": row["label"],
            "prediction": row["prediction"],
            "ai_probability": row["ai_probability"],
            "question": row["question"][:120],
            "text_preview": row["text"][:180],
        }
        for row in errors[:limit]
    ]


def escape_svg(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_v3_chart(metrics_rows: list[dict[str, str]]) -> None:
    selected_models = [
        "hc3_mixed_tfidf_logreg",
        "clean_aligned_mixed_tfidf_logreg",
        "casual_aug_mixed_tfidf_logreg",
        "rewrite_aligned_mixed_tfidf_logreg",
    ]
    selected_tests = [
        "hc3_chatgpt",
        "deepseek_original",
        "deepseek_casual_rewrite",
        "deepseek_adversarial_rewrite",
    ]
    labels = {
        "hc3_chatgpt": "HC3 ChatGPT",
        "deepseek_original": "DeepSeek original",
        "deepseek_casual_rewrite": "Casual rewrite",
        "deepseek_adversarial_rewrite": "Adversarial rewrite",
    }
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]
    lookup = {(r["model"], r["test_set"]): float(r["f1"]) for r in metrics_rows}
    width, height = 1320, 540
    left, top, chart_h = 82, 72, 320
    group_w = 300
    bar_w = 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="40" font-family="Arial, Microsoft YaHei" font-size="22" font-weight="700">V3 aligned benchmark: F1 across generators and rewrite attacks</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - 40}" y2="{top + chart_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#374151"/>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}%</text>')

    for gi, test in enumerate(selected_tests):
        gx = left + 62 + gi * group_w
        for mi, model in enumerate(selected_models):
            val = lookup.get((model, test), 0.0)
            h = chart_h * val
            x = gx + mi * (bar_w + 12)
            y = top + chart_h - h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="4" fill="{colors[mi]}"/>')
            parts.append(f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{val * 100:.0f}%</text>')
        parts.append(f'<text x="{gx + 92}" y="{top + chart_h + 30}" text-anchor="middle" font-family="Arial" font-size="13">{escape_svg(labels[test])}</text>')

    legend_y = 454
    for i, model in enumerate(selected_models):
        x = 86 + i * 300
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" rx="2" fill="{colors[i]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-family="Arial" font-size="12">{escape_svg(model)}</text>')
    parts.append("</svg>")
    (RESULTS_DIR / "v3_f1_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def write_dataset_overview(rows: list[dict[str, str]]) -> None:
    output = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["domain"])].append(row)
    for (split, domain), group in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "domain": domain,
                "questions": str(len(group)),
                "samples_if_all_variants": str(len(group) * 5),
                "human_samples": str(len(group)),
                "ai_samples": str(len(group) * 4),
            }
        )
    write_csv(
        RESULTS_DIR / "v3_dataset_overview.csv",
        output,
        ["split", "domain", "questions", "samples_if_all_variants", "human_samples", "ai_samples"],
    )


def write_top_features(model_name: str, weights: list[float], vocab: dict[str, int]) -> None:
    reverse = {idx: token for token, idx in vocab.items()}
    ranked = sorted(((weight, reverse[idx]) for idx, weight in enumerate(weights)), reverse=True)
    lines = [
        f"# Top Features: {model_name}",
        "",
        "Positive features push a sample toward the AI-generated class. Negative features push it toward the human class.",
        "",
        "## AI-Leaning Features",
        "",
        "| rank | token | weight |",
        "| ---: | --- | ---: |",
    ]
    for rank, (weight, token) in enumerate(ranked[:30], start=1):
        safe_token = token.replace("|", "\\|")
        lines.append(f"| {rank} | `{safe_token}` | {weight:.4f} |")
    lines.extend(["", "## Human-Leaning Features", "", "| rank | token | weight |", "| ---: | --- | ---: |"])
    for rank, (weight, token) in enumerate(reversed(ranked[-30:]), start=1):
        safe_token = token.replace("|", "\\|")
        lines.append(f"| {rank} | `{safe_token}` | {weight:.4f} |")
    (RESULTS_DIR / f"v3_top_features_{model_name}.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    metrics_rows: list[dict[str, str]],
    aligned_rows: list[dict[str, str]],
    model_notes: dict[str, str],
) -> None:
    rows = sorted(metrics_rows, key=lambda r: (r["model"], r["test_set"]))
    question_count = len(aligned_rows)
    domain_count = len({row["domain"] for row in aligned_rows})
    split_counts = Counter(row["split"] for row in aligned_rows)
    best_adv = max((r for r in rows if r["test_set"] == "deepseek_adversarial_rewrite"), key=lambda r: float(r["f1"]))
    hc3_only_adv = next(
        r
        for r in rows
        if r["model"] == "hc3_mixed_tfidf_logreg" and r["test_set"] == "deepseek_adversarial_rewrite"
    )
    full_rewrite_adv = next(
        r
        for r in rows
        if r["model"] == "rewrite_aligned_mixed_tfidf_logreg" and r["test_set"] == "deepseek_adversarial_rewrite"
    )
    improvement = float(full_rewrite_adv["f1"]) - float(hc3_only_adv["f1"])

    lines = [
        "# V3 Experiment Summary",
        "",
        "## What V3 Fixes",
        "",
        "- V2 used a large public HC3 subset plus a separate 36-question DeepSeek attack set.",
        "- V3 aligns generators on the same questions: every selected HC3 question has human, ChatGPT, DeepSeek original, casual rewrite, and adversarial rewrite answers.",
        "- The split is question-level, so evaluation questions are unseen during training and threshold calibration.",
        "",
        "## Dataset",
        "",
        f"- Aligned questions: {question_count}",
        f"- Domains: {domain_count}",
        f"- Total text samples if all variants are expanded: {question_count * 5}",
        f"- Train/calibration/eval questions: {split_counts['train']} / {split_counts['calibration']} / {split_counts['eval']}",
        "- Eval set size per test condition: 48 human answers + 48 AI answers.",
        "",
        "## Training Modes",
        "",
    ]
    for model, note in model_notes.items():
        lines.append(f"- `{model}`: {note}")

    lines.extend(
        [
            "",
            "## Main Results",
            "",
            "| model | test set | threshold | accuracy | precision | recall | F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['test_set']} | {float(row['threshold']):.2f} | "
            f"{float(row['accuracy']):.2%} | {float(row['precision']):.2%} | "
            f"{float(row['recall']):.2%} | {float(row['f1']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            f"- Best adversarial-rewrite model: `{best_adv['model']}` with F1={float(best_adv['f1']):.2%}.",
            f"- HC3-only mixed model adversarial F1: {float(hc3_only_adv['f1']):.2%}.",
            f"- Full aligned rewrite augmentation adversarial F1: {float(full_rewrite_adv['f1']):.2%}.",
            f"- Absolute adversarial F1 gain over HC3-only mixed baseline: {improvement:.2%}.",
            "- The experiment now measures generator shift and rewrite robustness under a controlled same-question setting.",
            "",
            "## Limitations",
            "",
            "- DeepSeek rewrites are prompt-generated and should be manually spot-checked before being treated as a formal benchmark.",
            "- Current detectors are lightweight TF-IDF baselines, not fine-tuned transformer detectors.",
            "- The dataset is still a compact research demo, but it is now large and controlled enough to support a credible interview discussion.",
            "",
        ]
    )
    (RESULTS_DIR / "v3_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Missing aligned dataset. Run scripts/generate_deepseek_aligned_hc3.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aligned_rows = load_csv(DATA_PATH)
    write_dataset_overview(aligned_rows)

    rows_by_split = {
        split: [row for row in aligned_rows if row["split"] == split]
        for split in ["train", "calibration", "eval"]
    }
    test_sets = {
        name: build_pair_records(rows_by_split["eval"], variants)
        for name, variants in TEST_SETS.items()
    }

    metrics_rows: list[dict[str, str]] = []
    prediction_rows: list[dict[str, str]] = []
    domain_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    model_notes = {spec["model"]: spec["note"] for spec in MODEL_SPECS}
    feature_artifacts: dict[str, tuple[list[float], dict[str, int]]] = {}

    for spec in MODEL_SPECS:
        model_name = spec["model"]
        feature_kind = spec["feature_kind"]
        learner = spec["learner"]
        train_ai_variants = spec["train_ai_variants"]
        train_rows = build_pair_records(rows_by_split["train"], train_ai_variants)
        cal_rows = build_pair_records(rows_by_split["calibration"], train_ai_variants)

        if feature_kind == "tfidf_char":
            tokenizer = char_ngrams
        elif feature_kind == "tfidf_mixed":
            tokenizer = mixed_tokens
        else:
            raise ValueError(f"Unsupported feature kind: {feature_kind}")

        vocab, idf = build_vocab(train_rows, tokenizer, min_df=2)
        vectorizer = lambda text, vocab=vocab, idf=idf, tokenizer=tokenizer: vectorize_tfidf(text, tokenizer, vocab, idf)
        dim = len(vocab)
        vectors = [vectorizer(row["text"]) for row in train_rows]
        labels = [int(row["label"]) for row in train_rows]

        if learner != "logreg":
            raise ValueError(f"Unsupported learner: {learner}")
        weights, bias = train_logreg(vectors, labels, dim)
        feature_artifacts[model_name] = (weights, vocab)
        threshold, cal_metrics = calibrate(cal_rows, vectorizer, weights, bias)

        for test_name, rows in test_sets.items():
            metrics, preds = evaluate(rows, vectorizer, weights, bias, threshold)
            metrics_rows.append(
                format_metric_row(
                    {
                        "model": model_name,
                        "feature_kind": feature_kind,
                        "learner": learner,
                        "training_variants": "+".join(train_ai_variants),
                        "train_samples": len(train_rows),
                        "calibration_samples": len(cal_rows),
                        "test_set": test_name,
                        "threshold": threshold,
                        "calibration_f1": cal_metrics["f1"],
                        **metrics,
                    }
                )
            )
            for pred in preds:
                prediction_rows.append({"model": model_name, "test_set": test_name, **pred})
            for row in domain_metrics(preds):
                domain_rows.append({"model": model_name, "test_set": test_name, **row})
            error_rows.extend(top_errors(preds, model_name, test_name, limit=20))

        print(f"trained {model_name}: vocab={dim}, train={len(train_rows)}, threshold={threshold:.2f}")

    write_csv(
        RESULTS_DIR / "v3_metrics.csv",
        metrics_rows,
        [
            "model",
            "feature_kind",
            "learner",
            "training_variants",
            "train_samples",
            "calibration_samples",
            "test_set",
            "threshold",
            "calibration_f1",
            "accuracy",
            "balanced_accuracy",
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
        RESULTS_DIR / "v3_predictions.csv",
        prediction_rows,
        [
            "model",
            "test_set",
            "sample_id",
            "question_id",
            "dataset",
            "domain",
            "split",
            "variant",
            "pair_variant",
            "label",
            "question",
            "text",
            "prediction",
            "ai_probability",
        ],
    )
    write_csv(
        RESULTS_DIR / "v3_metrics_by_domain.csv",
        domain_rows,
        [
            "model",
            "test_set",
            "domain",
            "accuracy",
            "balanced_accuracy",
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
        RESULTS_DIR / "v3_error_analysis.csv",
        error_rows,
        [
            "model",
            "test_set",
            "sample_id",
            "question_id",
            "domain",
            "variant",
            "pair_variant",
            "label",
            "prediction",
            "ai_probability",
            "question",
            "text_preview",
        ],
    )
    for model_name in ["hc3_mixed_tfidf_logreg", "rewrite_aligned_mixed_tfidf_logreg"]:
        weights, vocab = feature_artifacts[model_name]
        write_top_features(model_name, weights, vocab)
    write_v3_chart(metrics_rows)
    write_summary(metrics_rows, aligned_rows, model_notes)

    print("V3 experiment finished.")
    for row in metrics_rows:
        print(f"{row['model']} / {row['test_set']}: F1={float(row['f1']):.2%}, Acc={float(row['accuracy']):.2%}")


if __name__ == "__main__":
    main()
