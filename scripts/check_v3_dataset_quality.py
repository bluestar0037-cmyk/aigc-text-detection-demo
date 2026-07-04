"""Check basic quality statistics for the V3 aligned dataset."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hc3_deepseek_aligned_300.csv"
RESULTS_DIR = PROJECT_ROOT / "results_v3"
QUALITY_PATH = RESULTS_DIR / "v3_dataset_quality.json"
SUSPICIOUS_PATH = RESULTS_DIR / "v3_suspicious_samples.csv"

VARIANT_COLUMNS = [
    "human_hc3",
    "chatgpt_hc3",
    "deepseek_original",
    "deepseek_rewrite_casual",
    "deepseek_rewrite_adversarial",
]

DEEPSEEK_COLUMNS = [
    "deepseek_original",
    "deepseek_rewrite_casual",
    "deepseek_rewrite_adversarial",
]


def text_len(text: str) -> int:
    return len("".join(text.split()))


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Missing data/hc3_deepseek_aligned_300.csv")

    rows = load_rows()
    missing_cells = []
    lengths: dict[str, list[int]] = defaultdict(list)
    all_texts: Counter[str] = Counter()
    suspicious = []

    for row in rows:
        for column in VARIANT_COLUMNS:
            value = row.get(column, "")
            clean = value.strip()
            if not clean:
                missing_cells.append({"question_id": row["question_id"], "column": column})
                continue
            length = text_len(clean)
            lengths[column].append(length)
            all_texts[clean] += 1
            if column in DEEPSEEK_COLUMNS and (length < 40 or length > 260):
                suspicious.append(
                    {
                        "question_id": row["question_id"],
                        "domain": row["domain"],
                        "split": row["split"],
                        "column": column,
                        "length": str(length),
                        "question": row["question"],
                        "text_preview": clean[:180],
                    }
                )

    duplicate_texts = sum(1 for count in all_texts.values() if count > 1)
    split_counts = Counter(row["split"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    quality = {
        "rows": len(rows),
        "expanded_samples": len(rows) * len(VARIANT_COLUMNS),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "missing_cells": len(missing_cells),
        "exact_duplicate_text_values": duplicate_texts,
        "deepseek_length_outliers": len(suspicious),
        "length_stats": {column: summarize_lengths(values) for column, values in lengths.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        SUSPICIOUS_PATH,
        suspicious,
        ["question_id", "domain", "split", "column", "length", "question", "text_preview"],
    )

    print(f"Wrote {QUALITY_PATH}")
    print(f"Wrote {SUSPICIOUS_PATH}")
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
