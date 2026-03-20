#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PLANNER_DIR = SCRIPT_DIR.parent / "planner"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from deepseek_task_decomposer import DeepSeekTaskDecomposer, load_api_key
from planner_decomposition_utils import normalize_decomposition_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose RoboTwin task instructions with DeepSeek.")
    parser.add_argument("input_json", type=str, help="Path to the input JSON containing seen/unseen instruction lists.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--jsonl-output", type=str, default=None, help="Optional JSONL path for flattened records.")
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY", help="Environment variable containing the DeepSeek API key.")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible base URL.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for each instruction.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--deduplicate", action="store_true", help="Only call the API once for duplicate instructions.")
    return parser.parse_args()


def load_input(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for split in ("seen", "unseen"):
        payload.setdefault(split, [])
        if not isinstance(payload[split], list):
            raise TypeError(f"Expected list for split '{split}'.")
    return payload


def iter_records(payload: Dict[str, List[str]]) -> Iterable[Tuple[str, int, str]]:
    for split in ("seen", "unseen"):
        for idx, instruction in enumerate(payload.get(split, [])):
            yield split, idx, instruction


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_key_env)
    payload = load_input(args.input_json)

    decomposer = DeepSeekTaskDecomposer(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    cache: Dict[str, Dict[str, Any]] = OrderedDict()
    results: Dict[str, List[Dict[str, Any]]] = {"seen": [], "unseen": []}
    flat_rows: List[Dict[str, Any]] = []

    for split, idx, instruction in iter_records(payload):
        if args.deduplicate and instruction in cache:
            decomposition = cache[instruction]
        else:
            decomposition = decomposer.decompose_instruction(instruction)
            cache[instruction] = decomposition

        row = {
            "split": split,
            "index": idx,
            "instruction": instruction,
            "decomposition": decomposition,
        }
        row = normalize_decomposition_row(row, expand_stack_execution=True)
        results[split].append(row)
        flat_rows.append(row)
        print(f"[{split}] {idx + 1}: decomposed")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in flat_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": os.path.abspath(args.input_json),
        "output": str(output_path.resolve()),
        "jsonl_output": str(Path(args.jsonl_output).resolve()) if args.jsonl_output else None,
        "seen_count": len(results["seen"]),
        "unseen_count": len(results["unseen"]),
        "unique_instructions": len(cache),
        "deduplicate": args.deduplicate,
        "model": args.model,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
