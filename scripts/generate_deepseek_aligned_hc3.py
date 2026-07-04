"""Generate DeepSeek outputs aligned to the HC3-Chinese subset.

For each selected HC3 question, this script keeps:
- HC3 human answer
- HC3 ChatGPT answer
- DeepSeek original answer
- DeepSeek casual rewrite
- DeepSeek adversarial rewrite

The API key is read from DEEPSEEK_API_KEY and is never written to disk.
The script is resumable: existing rows in the output CSV are skipped.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HC3_PATH = PROJECT_ROOT / "data" / "hc3_chinese_public_subset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hc3_deepseek_aligned_300.csv"
USAGE_PATH = PROJECT_ROOT / "results_v3" / "deepseek_aligned_usage.json"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
QUESTIONS_PER_DOMAIN = int(os.environ.get("V3_QUESTIONS_PER_DOMAIN", "50"))
CONCURRENCY = int(os.environ.get("DEEPSEEK_CONCURRENCY", "4"))
SLEEP_SECONDS = float(os.environ.get("DEEPSEEK_SLEEP_SECONDS", "0.2"))
REGENERATE_SHORT_ROWS = os.environ.get("V3_REGENERATE_SHORT_ROWS", "0") == "1"
MIN_DEEPSEEK_CHARS = int(os.environ.get("V3_MIN_DEEPSEEK_CHARS", "40"))
MAX_DEEPSEEK_CHARS = int(os.environ.get("V3_MAX_DEEPSEEK_CHARS", "260"))


FIELDNAMES = [
    "question_id",
    "domain",
    "split",
    "question",
    "human_hc3",
    "chatgpt_hc3",
    "deepseek_original",
    "deepseek_rewrite_casual",
    "deepseek_rewrite_adversarial",
]


write_lock = threading.Lock()
usage_lock = threading.Lock()


def load_hc3_questions() -> list[dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    with HC3_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            item = grouped.setdefault(
                qid,
                {
                    "question_id": qid,
                    "domain": row["domain"],
                    "split": row["split"],
                    "question": row["question"],
                    "human_hc3": "",
                    "chatgpt_hc3": "",
                },
            )
            if row["variant"] == "human_hc3":
                item["human_hc3"] = row["text"]
            elif row["variant"] == "chatgpt_hc3":
                item["chatgpt_hc3"] = row["text"]

    by_domain_split: dict[tuple[str, str], list[dict[str, str]]] = {}
    for item in grouped.values():
        if not item["human_hc3"] or not item["chatgpt_hc3"]:
            continue
        by_domain_split.setdefault((item["domain"], item["split"]), []).append(item)

    selected: list[dict[str, str]] = []
    domains = sorted({item["domain"] for item in grouped.values()})
    for domain in domains:
        train = sorted(by_domain_split.get((domain, "train"), []), key=lambda x: x["question_id"])[:35]
        calibration = sorted(by_domain_split.get((domain, "calibration"), []), key=lambda x: x["question_id"])[:7]
        eval_rows = sorted(by_domain_split.get((domain, "eval"), []), key=lambda x: x["question_id"])[:8]
        domain_rows = (train + calibration + eval_rows)[:QUESTIONS_PER_DOMAIN]
        selected.extend(domain_rows)
    return selected


def load_existing() -> dict[str, dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return {}
    with OUTPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            if REGENERATE_SHORT_ROWS and row_needs_regeneration(row):
                continue
            rows[row["question_id"]] = row
        return rows


def compact_len(text: str) -> int:
    return len("".join(text.split()))


def row_needs_regeneration(row: dict[str, str]) -> bool:
    for column in ["deepseek_original", "deepseek_rewrite_casual", "deepseek_rewrite_adversarial"]:
        length = compact_len(row.get(column, ""))
        if length < MIN_DEEPSEEK_CHARS or length > MAX_DEEPSEEK_CHARS:
            return True
    return False


def write_dataset(rows: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["question_id"]))


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
    required = ["deepseek_original", "deepseek_rewrite_casual", "deepseek_rewrite_adversarial"]
    missing = [key for key in required if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return {key: " ".join(str(data[key]).split()) for key in required}


def call_deepseek(item: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    prompt = f"""
你正在构建中文 AIGC 文本检测鲁棒性评估数据。

领域：{item['domain']}
问题：{item['question']}

请生成三种中文回答：

1. deepseek_original：正常知识问答风格，表达完整、正式、条理清楚。
2. deepseek_rewrite_casual：保持意思基本一致，改写成普通大学生自然表达，减少模板化连接词。
3. deepseek_rewrite_adversarial：更强改写，目标是削弱 AI 文本检测器可能依赖的表层痕迹；更口语、更具体、句式更不规则，但不能改变事实含义。

要求：
- 每个字段 80 到 170 个中文字符左右。
- 如涉及医学、法律、金融等专业问题，请保持谨慎，不要给危险建议。
- 不要出现“我是AI”“作为模型”“生成文本”等字样。
- 只输出严格 JSON，不要 Markdown。
- JSON 键名必须为 deepseek_original、deepseek_rewrite_casual、deepseek_rewrite_adversarial。
""".strip()

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是严谨的中文数据生成助手，只输出合法 JSON。"},
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
        with urllib.request.urlopen(req, timeout=100) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {e.code}: {detail}") from e
    content = data["choices"][0]["message"]["content"]
    return extract_json(content), data.get("usage", {})


def generate_one(item: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            generated, usage = call_deepseek(item)
            return {
                "question_id": item["question_id"],
                "domain": item["domain"],
                "split": item["split"],
                "question": item["question"],
                "human_hc3": item["human_hc3"],
                "chatgpt_hc3": item["chatgpt_hc3"],
                **generated,
            }, usage
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed {item['question_id']}: {last_error}") from last_error


def update_usage(total_usage: dict[str, int], usage: dict[str, Any]) -> None:
    with usage_lock:
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            total_usage[key] = total_usage.get(key, 0) + int(usage.get(key, 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            total_usage["cached_tokens"] = total_usage.get("cached_tokens", 0) + int(details.get("cached_tokens", 0) or 0)


def write_usage(total_rows: int, new_rows: int, total_usage: dict[str, int]) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(
        json.dumps(
            {
                "model": MODEL,
                "target_rows": total_rows,
                "generated_new_rows_this_run": new_rows,
                "usage_for_new_rows_this_run": total_usage,
                "note": "Rows skipped from previous runs are not included in this run usage.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    selected = load_hc3_questions()
    existing = load_existing()
    rows = list(existing.values())
    remaining = [item for item in selected if item["question_id"] not in existing]
    total_usage: dict[str, int] = {}
    generated_count = 0

    print(f"selected={len(selected)}, existing={len(existing)}, remaining={len(remaining)}, concurrency={CONCURRENCY}")
    if not remaining:
        write_dataset(rows)
        write_usage(len(selected), 0, total_usage)
        print("Nothing to generate.")
        return

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_map = {executor.submit(generate_one, item): item for item in remaining}
        for future in as_completed(future_map):
            item = future_map[future]
            result, usage = future.result()
            with write_lock:
                rows = list(load_existing().values())
                current = {row["question_id"]: row for row in rows}
                current[result["question_id"]] = result
                write_dataset(list(current.values()))
            update_usage(total_usage, usage)
            generated_count += 1
            write_usage(len(selected), generated_count, total_usage)
            print(f"[{generated_count}/{len(remaining)}] generated {item['question_id']}")
            time.sleep(SLEEP_SECONDS)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {USAGE_PATH}")


if __name__ == "__main__":
    main()
