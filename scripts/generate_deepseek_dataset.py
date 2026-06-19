"""Generate a real DeepSeek-backed dataset for the AIGC text detection demo.

The API key must be provided through the environment variable
DEEPSEEK_API_KEY. The key is never written to disk.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_BANK_PATH = PROJECT_ROOT / "data" / "question_bank.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "deepseek_dataset.csv"
USAGE_PATH = PROJECT_ROOT / "results" / "deepseek_usage.json"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_QUESTIONS = int(os.environ.get("DEEPSEEK_MAX_QUESTIONS", "0") or "0")
SLEEP_SECONDS = float(os.environ.get("DEEPSEEK_SLEEP_SECONDS", "0.8"))


FIELDNAMES = [
    "question_id",
    "split_group",
    "domain",
    "question",
    "human_reference",
    "ai_original",
    "ai_rewrite_casual",
    "ai_rewrite_adversarial",
]


def load_question_bank() -> list[dict[str, str]]:
    with QUESTION_BANK_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if MAX_QUESTIONS > 0:
        rows = rows[:MAX_QUESTIONS]
    return rows


def load_existing() -> dict[str, dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return {}
    with OUTPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        return {row["question_id"]: row for row in csv.DictReader(f)}


def write_dataset(rows: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def extract_json(text: str) -> dict[str, str]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    required = ["ai_original", "ai_rewrite_casual", "ai_rewrite_adversarial"]
    missing = [key for key in required if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"DeepSeek response missing fields: {missing}")
    return {key: str(data[key]).strip().replace("\n", " ") for key in required}


def call_deepseek(question: str, domain: str) -> tuple[dict[str, str], dict[str, Any]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

    prompt = f"""
你正在帮助构建一个中文 AIGC 文本检测实验数据集。

问题领域：{domain}
问题：{question}

请围绕这个问题生成三种中文回答：

1. ai_original：典型大模型原始回答，表达完整、正式、条理清楚，可以有“首先/其次/因此/建议”等结构化痕迹。
2. ai_rewrite_casual：在保持意思基本一致的前提下，把回答改写得更像普通大学生自然写出的文字，减少模板化连接词，不要写成列表。
3. ai_rewrite_adversarial：更强的改写版本，目标是削弱 AI 文本检测器可能依赖的表层痕迹。要求更口语、更具体、句式更不规则，但不能改变事实含义。

要求：
- 每个字段 90 到 160 个中文字符左右。
- 不要提到“我是 AI”“作为模型”“生成文本”等字样。
- 不要输出 Markdown。
- 只输出严格 JSON，键名必须是 ai_original、ai_rewrite_casual、ai_rewrite_adversarial。
""".strip()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的中文数据生成助手，只输出用户要求的 JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.75,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {e.code}: {detail}") from e

    content = data["choices"][0]["message"]["content"]
    return extract_json(content), data.get("usage", {})


def update_usage(total_usage: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total_usage[key] = total_usage.get(key, 0) + int(usage.get(key, 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        total_usage["cached_tokens"] = total_usage.get("cached_tokens", 0) + int(details.get("cached_tokens", 0) or 0)


def main() -> None:
    PROJECT_ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
    question_rows = load_question_bank()
    existing = load_existing()
    output_rows: list[dict[str, str]] = []
    total_usage: dict[str, int] = {}

    for idx, row in enumerate(question_rows, 1):
        qid = row["question_id"]
        if qid in existing:
            output_rows.append(existing[qid])
            print(f"[skip] {qid} already exists")
            continue

        print(f"[generate] {idx}/{len(question_rows)} {qid} {row['domain']}")
        generated = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                generated, usage = call_deepseek(row["question"], row["domain"])
                update_usage(total_usage, usage)
                break
            except Exception as e:  # noqa: BLE001 - show retryable API details.
                last_error = e
                print(f"  attempt {attempt} failed: {e}")
                time.sleep(2.0 * attempt)
        if generated is None:
            raise RuntimeError(f"Failed to generate {qid}") from last_error

        output_row = {
            "question_id": qid,
            "split_group": row["split_group"],
            "domain": row["domain"],
            "question": row["question"],
            "human_reference": row["human_reference"],
            **generated,
        }
        output_rows.append(output_row)
        write_dataset(output_rows)
        USAGE_PATH.write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "generated_rows": len(output_rows),
                    "usage_for_new_rows_this_run": total_usage,
                    "note": "Token usage does not include rows skipped from a previous run.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        time.sleep(SLEEP_SECONDS)

    write_dataset(output_rows)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {USAGE_PATH}")


if __name__ == "__main__":
    main()
