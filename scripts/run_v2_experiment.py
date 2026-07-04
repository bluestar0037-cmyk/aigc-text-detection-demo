"""Run the v2 robustness experiment.

V2 adds:
- public HC3-Chinese data
- cross-dataset evaluation against DeepSeek
- multiple baselines: character TF-IDF LR, word/character TF-IDF LR,
  linear SVM, and stylometric logistic regression
- error analysis and robustness gap reporting
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC3_PATH = PROJECT_ROOT / "data" / "hc3_chinese_public_subset.csv"
DEEPSEEK_PATH = PROJECT_ROOT / "data" / "deepseek_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results_v2"

CONNECTIVES = [
    "首先",
    "其次",
    "此外",
    "因此",
    "总之",
    "建议",
    "需要",
    "可以",
    "应该",
    "如果",
    "同时",
    "最后",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def mixed_tokens(text: str) -> list[str]:
    clean = normalize(text)
    tokens = char_ngrams(text, 2, 4)
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", clean)
    tokens.extend(f"W:{chunk}" for chunk in chunks)
    return tokens


def stylometric_features(text: str) -> dict[str, float]:
    chars = [c for c in text if not c.isspace()]
    length = max(len(chars), 1)
    sentences = [s for s in re.split(r"[。！？!?]", text) if s.strip()]
    sent_count = max(len(sentences), 1)
    punctuation = sum(1 for c in text if c in "，。！？；：、,.!?;:")
    digits = sum(1 for c in text if c.isdigit())
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    unique_ratio = len(set(chars)) / length
    connective_count = sum(text.count(word) for word in CONNECTIVES)
    list_marker_count = len(re.findall(r"(^|\s)[0-9一二三四五六七八九十]+[、.．)]", text))
    return {
        "bias_feature": 1.0,
        "log_len": math.log(length + 1),
        "avg_sentence_len": length / sent_count,
        "punct_ratio": punctuation / length,
        "digit_ratio": digits / length,
        "ascii_ratio": ascii_chars / length,
        "unique_ratio": unique_ratio,
        "connective_per_100": 100 * connective_count / length,
        "list_marker_per_100": 100 * list_marker_count / length,
    }


def load_hc3_records() -> list[dict[str, str]]:
    records = []
    for row in load_csv(HC3_PATH):
        records.append(
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "dataset": "HC3-Chinese",
                "domain": row["domain"],
                "split": row["split"],
                "variant": row["variant"],
                "label": row["label"],
                "question": row["question"],
                "text": row["text"],
            }
        )
    return records


def load_deepseek_records() -> list[dict[str, str]]:
    records = []
    variant_map = {
        "human_reference": "human_reference",
        "ai_original": "deepseek_original",
        "ai_rewrite_casual": "deepseek_casual_rewrite",
        "ai_rewrite_adversarial": "deepseek_adversarial_rewrite",
    }
    for row in load_csv(DEEPSEEK_PATH):
        split = "eval" if row["split_group"] == "eval" else "train"
        base = {
            "question_id": row["question_id"],
            "dataset": "DeepSeek-Generated",
            "domain": row["domain"],
            "split": split,
            "question": row["question"],
        }
        records.append(
            {
                **base,
                "sample_id": f"{row['question_id']}_human_reference",
                "variant": "human_reference",
                "label": "0",
                "text": row["human_reference"],
            }
        )
        for column, variant in variant_map.items():
            if column == "human_reference":
                continue
            records.append(
                {
                    **base,
                    "sample_id": f"{row['question_id']}_{variant}",
                    "variant": variant,
                    "label": "1",
                    "text": row[column],
                }
            )
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


def build_stylo_vocab() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(stylometric_features("").keys())}


def vectorize_stylo(text: str, vocab: dict[str, int]) -> dict[int, float]:
    feats = stylometric_features(text)
    return {vocab[name]: value for name, value in feats.items()}


def dot(weights: list[float], features: dict[int, float]) -> float:
    return sum(weights[idx] * value for idx, value in features.items())


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def train_logreg(vectors: list[dict[int, float]], labels: list[int], dim: int, epochs=9, lr=0.45, l2=0.0005):
    weights = [0.0] * dim
    bias = 0.0
    for _ in range(epochs):
        for feats, label in zip(vectors, labels):
            pred = sigmoid(dot(weights, feats) + bias)
            error = pred - label
            for idx, value in feats.items():
                weights[idx] -= lr * (error * value + l2 * weights[idx])
            bias -= lr * error
    return weights, bias


def train_svm(vectors: list[dict[int, float]], labels: list[int], dim: int, epochs=12, lr=0.35, l2=0.0008):
    weights = [0.0] * dim
    bias = 0.0
    signed_labels = [1 if y == 1 else -1 for y in labels]
    for _ in range(epochs):
        for feats, label in zip(vectors, signed_labels):
            margin = label * (dot(weights, feats) + bias)
            if margin < 1:
                for idx, value in feats.items():
                    weights[idx] -= lr * (l2 * weights[idx] - label * value)
                bias += lr * label
            else:
                for idx in range(len(weights)):
                    weights[idx] -= lr * l2 * weights[idx]
    return weights, bias


def counts_to_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def evaluate(rows, vectorizer, weights, bias, threshold: float):
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


def calibrate(rows, vectorizer, weights, bias) -> tuple[float, dict[str, float]]:
    best = None
    for i in range(5, 96):
        threshold = i / 100
        metrics, _ = evaluate(rows, vectorizer, weights, bias, threshold)
        score = (metrics["f1"], metrics["accuracy"], metrics["recall"], metrics["precision"])
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def domain_metrics(predictions):
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
        m = counts_to_metrics(tp, fp, tn, fn)
        out.append({"domain": domain, **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in m.items()}})
    return out


def top_errors(predictions, model_name, test_set, limit=30):
    errors = [row for row in predictions if row["label"] != row["prediction"]]
    errors.sort(key=lambda row: abs(float(row["ai_probability"]) - 0.5), reverse=True)
    return [
        {
            "model": model_name,
            "test_set": test_set,
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "variant": row["variant"],
            "label": row["label"],
            "prediction": row["prediction"],
            "ai_probability": row["ai_probability"],
            "text_preview": row["text"][:160],
        }
        for row in errors[:limit]
    ]


def escape_svg(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_v2_chart(metrics_rows: list[dict[str, str]]) -> None:
    selected_tests = ["hc3_in_domain", "deepseek_original", "deepseek_adversarial_rewrite"]
    models = [
        "hc3_char_tfidf_logreg",
        "hc3_mixed_tfidf_logreg",
        "aug_char_tfidf_logreg",
        "aug_mixed_tfidf_logreg",
    ]
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    lookup = {(r["model"], r["test_set"]): float(r["f1"]) for r in metrics_rows}
    width, height = 1180, 500
    left, top, chart_h = 80, 70, 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="26" y="38" font-family="Arial, Microsoft YaHei" font-size="22" font-weight="700">V2 F1 comparison: public HC3 vs DeepSeek rewrite attacks</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - 40}" y2="{top + chart_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#374151"/>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}%</text>')

    group_w = 330
    bar_w = 42
    for gi, test in enumerate(selected_tests):
        gx = left + 70 + gi * group_w
        for mi, model in enumerate(models):
            val = lookup.get((model, test), 0.0)
            h = chart_h * val
            x = gx + mi * (bar_w + 14)
            y = top + chart_h - h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="4" fill="{colors[mi]}"/>')
            parts.append(f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{val * 100:.0f}%</text>')
        parts.append(f'<text x="{gx + 95}" y="{top + chart_h + 30}" text-anchor="middle" font-family="Arial" font-size="13">{escape_svg(test)}</text>')
    for i, model in enumerate(models):
        x = 90 + i * 260
        parts.append(f'<rect x="{x}" y="435" width="14" height="14" rx="2" fill="{colors[i]}"/>')
        parts.append(f'<text x="{x + 20}" y="447" font-family="Arial" font-size="12">{escape_svg(model)}</text>')
    parts.append("</svg>")
    (RESULTS_DIR / "v2_f1_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def summarize(metrics_rows, dataset_counts):
    rows = sorted(metrics_rows, key=lambda r: (r["model"], r["test_set"]))
    best_hc3 = max((r for r in rows if r["test_set"] == "hc3_in_domain"), key=lambda r: float(r["f1"]))
    best_attack = max((r for r in rows if r["test_set"] == "deepseek_adversarial_rewrite"), key=lambda r: float(r["f1"]))
    lines = [
        "# V2 Experiment Summary",
        "",
        "## What Changed From V1",
        "",
        "- Added the public HC3-Chinese dataset as the main benchmark.",
        "- Expanded from 36 custom questions to a 600-question public subset plus the existing DeepSeek attack set.",
        "- Added six baseline configurations and cross-dataset robustness evaluation.",
        "- Added domain-level metrics and high-confidence error analysis.",
        "",
        "## Dataset",
        "",
        f"- HC3-Chinese questions: {dataset_counts['hc3_questions']} ({dataset_counts['hc3_samples']} samples)",
        f"- DeepSeek attack questions: {dataset_counts['deepseek_questions']} ({dataset_counts['deepseek_samples']} samples)",
        "- HC3 fields: human answers vs ChatGPT answers.",
        "- DeepSeek fields: human reference vs original/casual/adversarial rewritten DeepSeek answers.",
        "",
        "## Main Results",
        "",
        "| model | test set | threshold | accuracy | precision | recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['test_set']} | {float(row['threshold']):.2f} | "
            f"{float(row['accuracy']):.2%} | {float(row['precision']):.2%} | "
            f"{float(row['recall']):.2%} | {float(row['f1']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
            f"- Best HC3 in-domain model: `{best_hc3['model']}` with F1={float(best_hc3['f1']):.2%}.",
            f"- Best DeepSeek adversarial-rewrite model: `{best_attack['model']}` with F1={float(best_attack['f1']):.2%}.",
            "- Models trained only on public HC3 ChatGPT style do not automatically solve DeepSeek rewritten text.",
            "- The experiment now reflects the RAID/M4 idea: report robustness across domains, generators, and attacks instead of only clean accuracy.",
            "",
            "## Limitations",
            "",
            "- The HC3 subset is deterministic but compact; it is not the full 12.9k-row dataset.",
            "- No pretrained transformer is fine-tuned yet; current models are CPU-friendly baselines.",
            "- DeepSeek adversarial rewrites are generated by prompting, not by a broad attack library.",
            "",
        ]
    )
    (RESULTS_DIR / "v2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not HC3_PATH.exists():
        raise SystemExit("Missing HC3 subset. Run: python scripts/build_hc3_dataset.py")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hc3 = load_hc3_records()
    deepseek = load_deepseek_records()

    hc3_train = [r for r in hc3 if r["split"] == "train"]
    hc3_cal = [r for r in hc3 if r["split"] == "calibration"]
    hc3_eval = [r for r in hc3 if r["split"] == "eval"]

    deep_train = [r for r in deepseek if r["split"] == "train"]
    deep_eval = [r for r in deepseek if r["split"] == "eval"]
    deep_original_eval = [r for r in deep_eval if r["variant"] in {"human_reference", "deepseek_original"}]
    deep_casual_eval = [r for r in deep_eval if r["variant"] in {"human_reference", "deepseek_casual_rewrite"}]
    deep_adv_eval = [r for r in deep_eval if r["variant"] in {"human_reference", "deepseek_adversarial_rewrite"}]

    train_augmented = hc3_train + [r for r in deep_train if r["variant"] in {"human_reference", "deepseek_original", "deepseek_casual_rewrite", "deepseek_adversarial_rewrite"}]
    cal_augmented = hc3_cal

    test_sets = {
        "hc3_in_domain": hc3_eval,
        "deepseek_original": deep_original_eval,
        "deepseek_casual_rewrite": deep_casual_eval,
        "deepseek_adversarial_rewrite": deep_adv_eval,
    }

    model_specs = [
        ("hc3_char_tfidf_logreg", "logreg", "tfidf_char", "hc3_only"),
        ("hc3_mixed_tfidf_logreg", "logreg", "tfidf_mixed", "hc3_only"),
        ("hc3_char_tfidf_svm", "svm", "tfidf_char", "hc3_only"),
        ("aug_char_tfidf_logreg", "logreg", "tfidf_char", "hc3_plus_deepseek"),
        ("aug_mixed_tfidf_logreg", "logreg", "tfidf_mixed", "hc3_plus_deepseek"),
        ("hc3_stylometric_logreg", "logreg", "stylometric", "hc3_only"),
    ]

    metrics_rows = []
    prediction_rows = []
    domain_rows = []
    error_rows = []

    for model_name, learner, feature_kind, train_mode in model_specs:
        if train_mode == "hc3_plus_deepseek":
            train_rows = train_augmented
        else:
            train_rows = hc3_train
        cal_rows = cal_augmented

        if feature_kind == "tfidf_char":
            tokenizer = char_ngrams
            vocab, idf = build_vocab(train_rows, tokenizer, min_df=2)
            vectorizer = lambda text, vocab=vocab, idf=idf: vectorize_tfidf(text, tokenizer, vocab, idf)
            dim = len(vocab)
        elif feature_kind == "tfidf_mixed":
            tokenizer = mixed_tokens
            vocab, idf = build_vocab(train_rows, tokenizer, min_df=2)
            vectorizer = lambda text, vocab=vocab, idf=idf: vectorize_tfidf(text, tokenizer, vocab, idf)
            dim = len(vocab)
        else:
            vocab = build_stylo_vocab()
            vectorizer = lambda text, vocab=vocab: vectorize_stylo(text, vocab)
            dim = len(vocab)

        vectors = [vectorizer(r["text"]) for r in train_rows]
        labels = [int(r["label"]) for r in train_rows]
        if learner == "svm":
            weights, bias = train_svm(vectors, labels, dim)
        else:
            weights, bias = train_logreg(vectors, labels, dim)

        threshold, cal_metrics = calibrate(cal_rows, vectorizer, weights, bias)

        for test_name, rows in test_sets.items():
            metrics, preds = evaluate(rows, vectorizer, weights, bias, threshold)
            metrics_rows.append(
                {
                    "model": model_name,
                    "feature_kind": feature_kind,
                    "learner": learner,
                    "train_mode": train_mode,
                    "test_set": test_name,
                    "threshold": f"{threshold:.2f}",
                    "calibration_f1": f"{cal_metrics['f1']:.4f}",
                    **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in metrics.items()},
                }
            )
            for pred in preds:
                prediction_rows.append({"model": model_name, "test_set": test_name, **pred})
            for row in domain_metrics(preds):
                domain_rows.append({"model": model_name, "test_set": test_name, **row})
            error_rows.extend(top_errors(preds, model_name, test_name, limit=20))

    write_csv(
        RESULTS_DIR / "v2_metrics.csv",
        metrics_rows,
        ["model", "feature_kind", "learner", "train_mode", "test_set", "threshold", "calibration_f1", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "total"],
    )
    write_csv(
        RESULTS_DIR / "v2_predictions.csv",
        prediction_rows,
        ["model", "test_set", "sample_id", "question_id", "dataset", "domain", "split", "variant", "label", "question", "text", "prediction", "ai_probability"],
    )
    write_csv(
        RESULTS_DIR / "v2_metrics_by_domain.csv",
        domain_rows,
        ["model", "test_set", "domain", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "total"],
    )
    write_csv(
        RESULTS_DIR / "v2_error_analysis.csv",
        error_rows,
        ["model", "test_set", "sample_id", "domain", "variant", "label", "prediction", "ai_probability", "text_preview"],
    )
    dataset_counts = {
        "hc3_questions": len({r["question_id"] for r in hc3}),
        "hc3_samples": len(hc3),
        "deepseek_questions": len({r["question_id"] for r in deepseek}),
        "deepseek_samples": len(deepseek),
    }
    summarize(metrics_rows, dataset_counts)
    write_v2_chart(metrics_rows)

    print("V2 experiment finished.")
    for row in metrics_rows:
        print(f"{row['model']} / {row['test_set']}: F1={float(row['f1']):.2%}, Acc={float(row['accuracy']):.2%}")


if __name__ == "__main__":
    main()
