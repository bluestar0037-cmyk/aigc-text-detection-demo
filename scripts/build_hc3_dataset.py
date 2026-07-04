"""Build a compact HC3-Chinese public benchmark subset.

The script downloads selected JSONL files from the public HuggingFace dataset
Hello-SimpleAI/HC3-Chinese and converts them into a flat binary-classification
CSV: human answer = 0, ChatGPT answer = 1.

Only a deterministic subset is kept in the committed CSV so the repository stays
small while still using public, citable data.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_hc3_chinese"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hc3_chinese_public_subset.csv"
SUMMARY_PATH = PROJECT_ROOT / "results_v2" / "hc3_dataset_summary.json"

BASE_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/main"
DOMAINS = ["finance", "law", "medicine", "nlpcc_dbqa", "open_qa", "psychology"]
MAX_QUESTIONS_PER_DOMAIN = 100
RANDOM_SEED = 20260704

FIELDNAMES = [
    "sample_id",
    "question_id",
    "dataset",
    "domain",
    "split",
    "variant",
    "label",
    "question",
    "text",
]


def download_if_needed(domain: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{domain}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = f"{BASE_URL}/{domain}.jsonl"
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r", "\n").split())


def first_answer(value) -> str:
    if isinstance(value, list):
        for item in value:
            text = clean_text(item)
            if text:
                return text
        return ""
    return clean_text(value)


def split_for_index(index: int, total: int) -> str:
    train_end = int(total * 0.70)
    calibration_end = int(total * 0.85)
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    return "eval"


def load_domain(domain: str) -> list[dict[str, str]]:
    path = download_if_needed(domain)
    questions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = clean_text(item.get("question", ""))
            human = first_answer(item.get("human_answers", []))
            chatgpt = first_answer(item.get("chatgpt_answers", []))
            if len(question) < 4 or len(human) < 20 or len(chatgpt) < 20:
                continue
            questions.append(
                {
                    "source_id": str(item.get("id", len(questions))),
                    "domain": domain,
                    "question": question,
                    "human": human,
                    "chatgpt": chatgpt,
                }
            )

    rng = random.Random(f"{RANDOM_SEED}-{domain}")
    rng.shuffle(questions)
    return questions[:MAX_QUESTIONS_PER_DOMAIN]


def main() -> None:
    rows: list[dict[str, str]] = []
    summary = {
        "dataset": "Hello-SimpleAI/HC3-Chinese",
        "source_url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
        "license": "cc-by-sa-4.0",
        "domains": {},
        "max_questions_per_domain": MAX_QUESTIONS_PER_DOMAIN,
        "random_seed": RANDOM_SEED,
    }

    for domain in DOMAINS:
        questions = load_domain(domain)
        summary["domains"][domain] = len(questions)
        for idx, item in enumerate(questions):
            split = split_for_index(idx, len(questions))
            question_id = f"hc3_{domain}_{idx:04d}"
            base = {
                "question_id": question_id,
                "dataset": "HC3-Chinese",
                "domain": domain,
                "split": split,
                "question": item["question"],
            }
            rows.append(
                {
                    **base,
                    "sample_id": f"{question_id}_human",
                    "variant": "human_hc3",
                    "label": "0",
                    "text": item["human"],
                }
            )
            rows.append(
                {
                    **base,
                    "sample_id": f"{question_id}_chatgpt",
                    "variant": "chatgpt_hc3",
                    "label": "1",
                    "text": item["chatgpt"],
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Questions: {len(rows) // 2}, samples: {len(rows)}")


if __name__ == "__main__":
    main()
