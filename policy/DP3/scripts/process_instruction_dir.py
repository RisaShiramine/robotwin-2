#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PLANNER_DIR = SCRIPT_DIR.parent / "planner"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from deepseek_task_decomposer import DeepSeekTaskDecomposer, load_api_key


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-process RoboTwin official per-episode instruction JSON files.")
    parser.add_argument("instruction_dir", type=str, help="Directory containing episode*.json instruction files.")
    parser.add_argument("--aggregated-output", required=True, help="Output aggregated seen/unseen instruction JSON.")
    parser.add_argument("--episode-map-output", required=True, help="Output per-episode chosen instruction map JSON.")
    parser.add_argument("--decomposition-input-output", default=None, help="Optional JSON file containing the exact instruction batch that will be used for decomposition.")
    parser.add_argument("--decomposition-output", default=None, help="Optional decomposition JSON output path.")
    parser.add_argument("--decomposition-jsonl-output", default=None, help="Optional decomposition JSONL output path.")
    parser.add_argument("--instruction-source", default="seen_first", choices=["seen", "unseen", "seen_first", "unseen_first"], help="How to choose the primary instruction for each episode in the episode map.")
    parser.add_argument("--decompose-scope", default="episode_map", choices=["episode_map", "all"], help="Whether to decompose only the chosen per-episode instruction set or all aggregated seen/unseen instructions.")
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY", help="Environment variable containing the DeepSeek API key.")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible base URL.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for each instruction.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--deduplicate", action="store_true", help="Only decompose each unique instruction once.")
    parser.add_argument("--normalize-dedup", action="store_true", help="Deduplicate using a normalized text key instead of exact-string matching.")
    parser.add_argument("--decompose", action="store_true", help="Call DeepSeek and write decomposition outputs after aggregation.")
    return parser.parse_args()


def normalize_text_key(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", text.strip().lower()).strip()


def choose_instruction(payload: Dict[str, List[str]], source: str) -> Tuple[str, str]:
    seen = payload.get("seen", [])
    unseen = payload.get("unseen", [])
    order = {
        "seen": [("seen", seen)],
        "unseen": [("unseen", unseen)],
        "seen_first": [("seen", seen), ("unseen", unseen)],
        "unseen_first": [("unseen", unseen), ("seen", seen)],
    }[source]
    for split_name, candidates in order:
        if candidates:
            return candidates[0], split_name
    raise ValueError("No instruction candidates found.")


def collect_instruction_dir(instruction_dir: str, instruction_source: str) -> Dict[str, Any]:
    root = Path(instruction_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Instruction directory does not exist: {instruction_dir}")

    seen_unique = OrderedDict()
    unseen_unique = OrderedDict()
    episode_map: Dict[str, Dict[str, Any]] = OrderedDict()

    for json_file in sorted(root.glob("episode*.json")):
        episode_id = int(json_file.stem.replace("episode", ""))
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        seen = payload.get("seen", [])
        unseen = payload.get("unseen", [])
        for instruction in seen:
            seen_unique.setdefault(instruction, None)
        for instruction in unseen:
            unseen_unique.setdefault(instruction, None)

        chosen_instruction, chosen_split = choose_instruction(payload, instruction_source)
        episode_map[str(episode_id)] = {
            "instruction": chosen_instruction,
            "instruction_split": chosen_split,
            "instruction_source": instruction_source,
            "seen_count": len(seen),
            "unseen_count": len(unseen),
            "seen": seen,
            "unseen": unseen,
        }

    return {
        "aggregated": {
            "seen": list(seen_unique.keys()),
            "unseen": list(unseen_unique.keys()),
        },
        "episode_map": episode_map,
    }


def build_decomposition_input(collected: Dict[str, Any], scope: str) -> Dict[str, List[str]]:
    if scope == "all":
        return collected["aggregated"]

    seen = OrderedDict()
    unseen = OrderedDict()
    for episode_meta in collected["episode_map"].values():
        split_name = episode_meta["instruction_split"]
        instruction = episode_meta["instruction"]
        if split_name == "seen":
            seen.setdefault(instruction, None)
        else:
            unseen.setdefault(instruction, None)
    return {
        "seen": list(seen.keys()),
        "unseen": list(unseen.keys()),
    }


def write_json(path: str, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def decompose_aggregated(
    aggregated: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, Any]]]:
    api_key = load_api_key(args.api_key_env)
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
    for split in ("seen", "unseen"):
        for idx, instruction in enumerate(aggregated.get(split, [])):
            cache_key = normalize_text_key(instruction) if args.normalize_dedup else instruction
            if args.deduplicate and cache_key in cache:
                decomposition = cache[cache_key]
            else:
                decomposition = decomposer.decompose_instruction(instruction)
                cache[cache_key] = decomposition
            row = {
                "split": split,
                "index": idx,
                "instruction": instruction,
                "decomposition": decomposition,
            }
            results[split].append(row)
            print(f"[{split}] {idx + 1}: decomposed")
    return results


def main() -> None:
    args = parse_args()
    collected = collect_instruction_dir(args.instruction_dir, args.instruction_source)
    decomposition_input = build_decomposition_input(collected, args.decompose_scope)

    write_json(args.aggregated_output, collected["aggregated"])
    write_json(args.episode_map_output, collected["episode_map"])
    if args.decomposition_input_output:
        write_json(args.decomposition_input_output, decomposition_input)

    summary = {
        "instruction_dir": str(Path(args.instruction_dir).resolve()),
        "aggregated_output": str(Path(args.aggregated_output).resolve()),
        "episode_map_output": str(Path(args.episode_map_output).resolve()),
        "decomposition_input_output": str(Path(args.decomposition_input_output).resolve()) if args.decomposition_input_output else None,
        "num_episodes": len(collected["episode_map"]),
        "num_seen_unique": len(collected["aggregated"]["seen"]),
        "num_unseen_unique": len(collected["aggregated"]["unseen"]),
        "decompose_scope": args.decompose_scope,
        "decompose_seen_count": len(decomposition_input["seen"]),
        "decompose_unseen_count": len(decomposition_input["unseen"]),
        "decompose": args.decompose,
    }

    if args.decompose:
        if not args.decomposition_output:
            raise ValueError("--decomposition-output is required when --decompose is set.")
        decompositions = decompose_aggregated(decomposition_input, args)
        write_json(args.decomposition_output, decompositions)
        summary["decomposition_output"] = str(Path(args.decomposition_output).resolve())
        if args.decomposition_jsonl_output:
            rows = decompositions["seen"] + decompositions["unseen"]
            output_path = Path(args.decomposition_jsonl_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["decomposition_jsonl_output"] = str(output_path.resolve())

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
