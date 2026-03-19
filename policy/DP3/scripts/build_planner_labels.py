#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TABLE_SUPPORTS = {None, "table", "center", "table_center"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert task decompositions into per-timestep planner labels.")
    parser.add_argument("--decomposition-file", required=True, help="JSON or JSONL produced by decompose_tasks.py.")
    parser.add_argument("--output", required=True, help="Output planner_labels.jsonl path.")
    parser.add_argument("--episode-instruction-map", default=None, help="JSON mapping episode ids to the instruction string or a richer metadata dict used in that episode.")
    parser.add_argument("--instruction-dir", default=None, help="Directory containing episode{idx}.json instruction files.")
    parser.add_argument("--instruction-source", default="seen_first", choices=["seen", "unseen", "seen_first", "unseen_first"], help="How to choose an instruction when reading per-episode instruction files.")
    parser.add_argument("--episode-lengths-json", default=None, help="JSON file mapping episode ids to transition lengths.")
    parser.add_argument("--stage-boundaries-json", default=None, help="Optional JSON file mapping episode ids to explicit stage ranges.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    parser.add_argument("--disable-stage-normalization", action="store_true", help="Use decomposition stages as-is without stack-stage normalization.")
    return parser.parse_args()


def normalize_text_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().lower().split())
    return text or None


def load_decomposition_file(path: str) -> List[Dict[str, Any]]:
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
    return rows


def build_decomposition_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_instruction: Dict[str, Dict[str, Any]] = {}
    by_canonical_task: Dict[str, Dict[str, Any]] = {}
    by_task_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        decomposition = row["decomposition"]
        instruction_key = normalize_text_key(row.get("instruction") or decomposition.get("instruction"))
        canonical_key = normalize_text_key(decomposition.get("canonical_task"))
        task_id = decomposition.get("task_id")
        if instruction_key:
            by_instruction[instruction_key] = row
        if canonical_key:
            by_canonical_task[canonical_key] = row
        if task_id is not None:
            by_task_id[str(task_id)] = row
    return {
        "instruction": by_instruction,
        "canonical_task": by_canonical_task,
        "task_id": by_task_id,
    }


def load_episode_instruction_map(path: str) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {idx: {"instruction": item} if isinstance(item, str) else item for idx, item in enumerate(payload)}
    normalized: Dict[int, Dict[str, Any]] = {}
    for k, v in payload.items():
        if isinstance(v, str):
            normalized[int(k)] = {"instruction": v}
        else:
            normalized[int(k)] = dict(v)
    return normalized


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


def load_instruction_dir(path: str, source: str) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for json_file in sorted(Path(path).glob("episode*.json")):
        episode_id = int(json_file.stem.replace("episode", ""))
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        mapping[episode_id] = {"instruction": choose_instruction(payload, source)}
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


def is_table_support(value: Any) -> bool:
    return value in TABLE_SUPPORTS


def infer_stage_type(stage: Dict[str, Any]) -> str:
    if stage.get("stage_type"):
        return stage["stage_type"]
    target_support = stage.get("target_support")
    target_location = stage.get("target_location")
    action_type = stage.get("action_type")
    if action_type == "stack" or (target_support and not is_table_support(target_support)):
        return "stack_on_support"
    if target_location:
        return "move_to_region"
    return action_type or "other"


def infer_source_object(stage: Dict[str, Any]) -> Optional[str]:
    return stage.get("source_object") or stage.get("target_object")


def infer_support_object(stage: Dict[str, Any]) -> Optional[str]:
    support = stage.get("support_object") or stage.get("target_support")
    return None if is_table_support(support) else support


def infer_target_region(stage: Dict[str, Any]) -> Optional[str]:
    return stage.get("target_region") or stage.get("target_location")


def build_stage_label(stage: Dict[str, Any]) -> str:
    source_object = infer_source_object(stage)
    support_object = infer_support_object(stage)
    target_region = infer_target_region(stage)
    if support_object:
        return f"{source_object} on {support_object}"
    if target_region:
        return f"{source_object} at {target_region}"
    action_type = stage.get("action_type") or stage.get("stage_type") or "stage"
    return f"{action_type} {source_object}".strip()


def topologically_order_stack_stages(stack_stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = []
    placed = set()
    remaining = stack_stages[:]
    while remaining:
        progress = False
        for stage in list(remaining):
            support = infer_support_object(stage)
            if support is None or support in placed:
                ordered.append(stage)
                placed.add(infer_source_object(stage))
                remaining.remove(stage)
                progress = True
        if not progress:
            ordered.extend(remaining)
            break
    return ordered


def normalize_decomposition_row(row: Dict[str, Any]) -> Dict[str, Any]:
    decomposition = row["decomposition"]
    stages = decomposition.get("stages", [])
    if decomposition.get("task_category") != "stack" or len(stages) <= 1:
        return row

    stack_stages = [stage for stage in stages if infer_support_object(stage) is not None]
    if not stack_stages:
        return row

    ordered_stack_stages = topologically_order_stack_stages(stack_stages)
    base_object = infer_support_object(ordered_stack_stages[0])
    base_stage = None
    for stage in stages:
        if infer_source_object(stage) == base_object and infer_support_object(stage) is None:
            base_stage = dict(stage)
            break

    if base_stage is None:
        base_stage = {
            "stage": f"place_{str(base_object).replace(' ', '_')}_at_center",
            "action_type": "place",
            "stage_type": "establish_base",
            "source_object": base_object,
            "support_object": None,
            "target_object": base_object,
            "target_support": "table",
            "target_location": "center",
            "target_region": "center",
            "spatial_relation": "on",
            "preferred_arm": None,
            "required_objects": [base_object],
            "completed_subgoals_before_stage": [],
            "success_criteria": f"{base_object} is placed at the center of the table.",
        }

    normalized_stages = [base_stage]
    normalized_stages.extend(dict(stage) for stage in ordered_stack_stages)

    completed = []
    for idx, stage in enumerate(normalized_stages):
        stage.setdefault("stage_type", infer_stage_type(stage))
        stage.setdefault("source_object", infer_source_object(stage))
        stage.setdefault("support_object", infer_support_object(stage))
        stage.setdefault("target_region", infer_target_region(stage))
        stage["completed_subgoals_before_stage"] = completed[:]
        completed.append(build_stage_label(stage))

    normalized = dict(row)
    normalized["decomposition"] = dict(decomposition)
    normalized["decomposition"]["stages"] = normalized_stages
    return normalized


def match_decomposition_row(index: Dict[str, Dict[str, Dict[str, Any]]], episode_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_id = episode_meta.get("task_id")
    if task_id is not None:
        row = index["task_id"].get(str(task_id))
        if row is not None:
            return row

    canonical_task = episode_meta.get("canonical_task")
    if canonical_task:
        row = index["canonical_task"].get(normalize_text_key(canonical_task))
        if row is not None:
            return row

    instruction = episode_meta.get("instruction")
    if instruction:
        return index["instruction"].get(normalize_text_key(instruction))
    return None


def classify_relevance(stage: Dict[str, Any], decomposition: Dict[str, Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    primary = dedupe_keep_order([
        infer_source_object(stage),
        infer_support_object(stage),
        infer_target_region(stage),
    ])
    secondary = dedupe_keep_order(stage.get("required_objects", []))
    distractors = [obj for obj in decomposition.get("scene_objects", []) if obj not in primary and obj not in secondary]
    return primary, secondary, distractors


def build_label_rows(
    episode_id: int,
    episode_meta: Dict[str, Any],
    decomposition_row: Dict[str, Any],
    episode_length: int,
    flat_index_start: int,
    stage_ranges: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    decomposition = decomposition_row["decomposition"]
    stages = decomposition["stages"]
    rows: List[Dict[str, Any]] = []
    instruction = episode_meta.get("instruction") or decomposition_row.get("instruction")

    for stage_index, (stage, (start, end)) in enumerate(zip(stages, stage_ranges)):
        completed_before = stage.get("completed_subgoals_before_stage", []) or [
            build_stage_label(previous_stage) for previous_stage in stages[:stage_index]
        ]
        primary, secondary, distractors = classify_relevance(stage, decomposition)
        relevance_objects = dedupe_keep_order(primary + secondary)
        for timestep in range(start, end):
            rows.append(
                {
                    "flat_index": flat_index_start + timestep,
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "instruction": instruction,
                    "canonical_task": decomposition.get("canonical_task"),
                    "task_id": decomposition.get("task_id", episode_meta.get("task_id")),
                    "split": decomposition_row.get("split"),
                    "source_index": decomposition_row.get("index"),
                    "task_category": decomposition.get("task_category"),
                    "final_goal": decomposition.get("final_goal"),
                    "stage": stage.get("stage"),
                    "stage_type": infer_stage_type(stage),
                    "stage_index": stage_index,
                    "num_stages": len(stages),
                    "is_terminal_stage": stage_index == len(stages) - 1,
                    "source_object": infer_source_object(stage),
                    "target_object": stage.get("target_object"),
                    "support_object": infer_support_object(stage),
                    "target_support": stage.get("target_support"),
                    "target_region": infer_target_region(stage),
                    "target_location": stage.get("target_location"),
                    "spatial_relation": stage.get("spatial_relation"),
                    "preferred_arm": stage.get("preferred_arm"),
                    "required_objects": stage.get("required_objects", []),
                    "completed_subgoals": completed_before,
                    "primary_objects": primary,
                    "secondary_objects": secondary,
                    "distractor_objects": distractors,
                    "relevance_objects": relevance_objects,
                    "success_criteria": stage.get("success_criteria"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    decomposition_rows = load_decomposition_file(args.decomposition_file)
    decomposition_index = build_decomposition_index(decomposition_rows)
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
    missing_instruction_matches: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for episode_id in sorted(episode_instruction_map):
            if episode_id not in episode_lengths:
                raise KeyError(f"Missing episode length for episode {episode_id}.")
            episode_meta = episode_instruction_map[episode_id]
            row = match_decomposition_row(decomposition_index, episode_meta)
            if row is None:
                missing_instruction_matches.append({"episode_id": episode_id, **episode_meta})
                continue
            if not args.disable_stage_normalization:
                row = normalize_decomposition_row(row)
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
                episode_meta=episode_meta,
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
        "missing_instruction_matches": missing_instruction_matches,
        "instruction_source": args.instruction_source if args.instruction_dir else "episode_instruction_map",
        "alignment_mode": "explicit_stage_boundaries" if stage_boundaries else "uniform_partition",
        "stage_normalization": not args.disable_stage_normalization,
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
