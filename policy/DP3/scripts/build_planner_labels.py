#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert task decompositions into per-timestep planner labels.")
    parser.add_argument("--decomposition-file", required=True, help="JSON or JSONL produced by decompose_tasks.py.")
    parser.add_argument("--output", required=True, help="Output planner_labels.jsonl path.")
    parser.add_argument("--episode-instruction-map", default=None, help="JSON mapping episode ids to the instruction string used in that episode.")
    parser.add_argument("--instruction-dir", default=None, help="Directory containing episode{idx}.json instruction files.")
    parser.add_argument("--instruction-source", default="seen_first", choices=["seen", "unseen", "seen_first", "unseen_first"], help="How to choose an instruction when reading per-episode instruction files.")
    parser.add_argument("--episode-lengths-json", default=None, help="JSON file mapping episode ids to transition lengths.")
    parser.add_argument("--stage-boundaries-json", default=None, help="Optional JSON file mapping episode ids to explicit stage ranges.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def load_decomposition_file(path: str) -> Dict[str, Dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        rows = []
        if isinstance(payload, dict) and ("seen" in payload or "unseen" in payload):
            for split in ("seen", "unseen"):
                rows.extend(payload.get(split, []))
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unsupported decomposition file format.")

    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        instruction = row["instruction"]
        mapping[instruction] = row
    return mapping


def load_episode_instruction_map(path: str) -> Dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {idx: instruction for idx, instruction in enumerate(payload)}
    return {int(k): v for k, v in payload.items()}


def choose_instruction(instruction_payload: Dict[str, List[str]], source: str) -> str:
    seen = instruction_payload.get("seen", [])
    unseen = instruction_payload.get("unseen", [])
    order = {
        "seen": [seen],
        "unseen": [unseen],
        "seen_first": [seen, unseen],
        "unseen_first": [unseen, seen],
    }[source]
    for candidates in order:
        if candidates:
            return candidates[0]
    raise ValueError("No instruction candidates found in instruction payload.")


def load_instruction_dir(path: str, source: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for json_file in sorted(Path(path).glob("episode*.json")):
        stem = json_file.stem
        episode_id = int(stem.replace("episode", ""))
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        mapping[episode_id] = choose_instruction(payload, source)
    return mapping


def load_episode_lengths(path: str) -> Dict[int, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {idx: int(length) for idx, length in enumerate(payload)}
    return {int(k): int(v) for k, v in payload.items()}


def load_stage_boundaries(path: str) -> Dict[int, List[Dict[str, int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload.items()}


def evenly_partition(length: int, num_stages: int) -> List[Tuple[int, int]]:
    if num_stages <= 0:
        raise ValueError("num_stages must be positive.")
    base = length // num_stages
    remainder = length % num_stages
    parts = []
    cursor = 0
    for idx in range(num_stages):
        width = base + (1 if idx < remainder else 0)
        start = cursor
        end = cursor + width
        parts.append((start, end))
        cursor = end
    return parts


def normalize_stage_boundaries(stage_ranges: List[Dict[str, int]], num_stages: int, length: int) -> List[Tuple[int, int]]:
    if len(stage_ranges) != num_stages:
        raise ValueError("Stage boundary count does not match decomposition stage count.")
    normalized = []
    for item in stage_ranges:
        start = int(item["start"])
        end = int(item["end"])
        if not (0 <= start < end <= length):
            raise ValueError(f"Invalid stage range {item} for episode length {length}.")
        normalized.append((start, end))
    return normalized


def dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for item in items:
        if item in (None, "", []):
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_label_rows(
    episode_id: int,
    instruction: str,
    decomposition_row: Dict[str, Any],
    episode_length: int,
    flat_index_start: int,
    stage_ranges: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    decomposition = decomposition_row["decomposition"]
    stages = decomposition["stages"]
    rows: List[Dict[str, Any]] = []

    for stage_index, (stage, (start, end)) in enumerate(zip(stages, stage_ranges)):
        completed_before = stage.get("completed_subgoals_before_stage", []) or [
            previous_stage["stage"] for previous_stage in stages[:stage_index]
        ]
        relevance_objects = dedupe_keep_order(
            list(stage.get("required_objects", []))
            + [stage.get("target_object"), stage.get("target_support"), stage.get("target_location")]
            + list(decomposition.get("scene_objects", []))
        )
        for timestep in range(start, end):
            rows.append(
                {
                    "flat_index": flat_index_start + timestep,
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "instruction": instruction,
                    "split": decomposition_row.get("split"),
                    "source_index": decomposition_row.get("index"),
                    "canonical_task": decomposition.get("canonical_task"),
                    "task_category": decomposition.get("task_category"),
                    "final_goal": decomposition.get("final_goal"),
                    "stage": stage.get("stage"),
                    "stage_index": stage_index,
                    "num_stages": len(stages),
                    "target_object": stage.get("target_object"),
                    "target_support": stage.get("target_support"),
                    "target_location": stage.get("target_location"),
                    "spatial_relation": stage.get("spatial_relation"),
                    "preferred_arm": stage.get("preferred_arm"),
                    "required_objects": stage.get("required_objects", []),
                    "completed_subgoals": completed_before,
                    "relevance_objects": relevance_objects,
                    "success_criteria": stage.get("success_criteria"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    decomposition_map = load_decomposition_file(args.decomposition_file)
    if args.episode_instruction_map:
        episode_instruction_map = load_episode_instruction_map(args.episode_instruction_map)
    elif args.instruction_dir:
        episode_instruction_map = load_instruction_dir(args.instruction_dir, args.instruction_source)
    else:
        raise ValueError("Either --episode-instruction-map or --instruction-dir must be provided.")

    if not args.episode_lengths_json:
        raise ValueError("--episode-lengths-json is currently required.")
    episode_lengths = load_episode_lengths(args.episode_lengths_json)
    stage_boundaries = load_stage_boundaries(args.stage_boundaries_json) if args.stage_boundaries_json else {}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat_index = 0
    total_rows = 0
    episodes_written = 0
    missing_instructions: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for episode_id in sorted(episode_instruction_map):
            if episode_id not in episode_lengths:
                raise KeyError(f"Missing episode length for episode {episode_id}.")
            instruction = episode_instruction_map[episode_id]
            row = decomposition_map.get(instruction)
            if row is None:
                missing_instructions.append({"episode_id": episode_id, "instruction": instruction})
                continue
            episode_length = episode_lengths[episode_id]
            stages = row["decomposition"]["stages"]
            if not stages:
                raise ValueError(f"No stages found for episode {episode_id}.")
            if episode_id in stage_boundaries:
                stage_ranges = normalize_stage_boundaries(stage_boundaries[episode_id], len(stages), episode_length)
            else:
                stage_ranges = evenly_partition(episode_length, len(stages))

            label_rows = build_label_rows(
                episode_id=episode_id,
                instruction=instruction,
                decomposition_row=row,
                episode_length=episode_length,
                flat_index_start=flat_index,
                stage_ranges=stage_ranges,
            )
            for label in label_rows:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")

            flat_index += episode_length
            total_rows += len(label_rows)
            episodes_written += 1

    summary = {
        "decomposition_file": str(Path(args.decomposition_file).resolve()),
        "output": str(output_path.resolve()),
        "episodes_requested": len(episode_instruction_map),
        "episodes_written": episodes_written,
        "rows_written": total_rows,
        "missing_instruction_matches": missing_instructions,
        "instruction_source": args.instruction_source if args.instruction_dir else "episode_instruction_map",
        "alignment_mode": "explicit_stage_boundaries" if stage_boundaries else "uniform_partition",
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
