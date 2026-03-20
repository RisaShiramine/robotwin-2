#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from planner_decomposition_utils import (
    build_stage_label,
    dedupe_keep_order,
    infer_source_object,
    infer_stage_type,
    infer_support_object,
    infer_target_region,
    normalize_decomposition_row,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert task decompositions into per-timestep planner labels.")
    parser.add_argument("--decomposition-file", required=True, help="JSON or JSONL produced by decompose_tasks.py.")
    parser.add_argument("--output", required=True, help="Output planner_labels.jsonl path.")
    parser.add_argument("--episode-instruction-map", default=None, help="JSON mapping episode ids to the instruction string or a richer metadata dict used in that episode.")
    parser.add_argument("--instruction-dir", default=None, help="Directory containing episode{idx}.json instruction files.")
    parser.add_argument("--instruction-source", default="seen_first", choices=["seen", "unseen", "seen_first", "unseen_first"], help="How to choose an instruction when reading per-episode instruction files.")
    parser.add_argument("--episode-lengths-json", default=None, help="JSON file mapping episode ids to transition lengths. Missing episodes can be filled from --dp3-zarr.")
    parser.add_argument("--dp3-zarr", default=None, help="Optional DP3 zarr path used to infer episode lengths from meta/episode_ends.")
    parser.add_argument("--stage-boundaries-json", default=None, help="Optional JSON file mapping episode ids to explicit stage ranges.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    parser.add_argument("--disable-stage-normalization", action="store_true", help="Use decomposition stages as-is without stack-stage normalization.")
    parser.add_argument("--boundary-mode", default="zarr_velocity", choices=["zarr_velocity", "uniform"], help="How to infer stage ranges when explicit boundaries are not provided.")
    parser.add_argument("--velocity-threshold", type=float, default=0.0, help="Optional explicit stationary threshold on ||action - state|| used for zarr-driven stage boundary inference. <=0 uses an adaptive threshold.")
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

def load_episode_lengths_from_zarr(zarr_path: str) -> Dict[int, int]:
    episode_ends = load_episode_ends_from_zarr(zarr_path)
    lengths: Dict[int, int] = {}
    previous_end = 0
    for episode_id, end in enumerate(episode_ends):
        end = int(end)
        lengths[episode_id] = end - previous_end
        previous_end = end
    return lengths

def load_episode_ends_from_zarr(zarr_path: str) -> List[int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "zarr is required to inspect --dp3-zarr. Install it or provide --episode-lengths-json."
        ) from exc

    root = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    return [int(value) for value in root["meta"]["episode_ends"][:]]

def merge_episode_lengths(explicit_lengths: Dict[int, int], zarr_lengths: Dict[int, int]) -> Dict[int, int]:
    merged = dict(zarr_lengths)
    merged.update(explicit_lengths)
    return merged

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

def load_episode_state_action_from_zarr(zarr_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    import zarr

    root = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    state = np.asarray(root["data"]["state"][:], dtype=np.float64)
    action = np.asarray(root["data"]["action"][:], dtype=np.float64)
    episode_ends = [int(value) for value in root["meta"]["episode_ends"][:]]

    episode_data: Dict[int, Dict[str, np.ndarray]] = {}
    start = 0
    for episode_id, end in enumerate(episode_ends):
        episode_data[episode_id] = {
            "state": state[start:end],
            "action": action[start:end],
        }
        start = end
    return episode_data


def _choose_stationary_threshold(velocity_norm: np.ndarray, explicit_threshold: float) -> float:
    if explicit_threshold > 0:
        return explicit_threshold
    positive = velocity_norm[velocity_norm > 0]
    if positive.size == 0:
        return 1e-6
    floor = float(np.percentile(positive, 10))
    ceiling = float(np.percentile(positive, 35))
    return max(1e-6, min(floor * 1.5, ceiling))


def _find_stationary_centers(velocity_norm: np.ndarray, threshold: float) -> List[int]:
    stationary = velocity_norm <= threshold
    centers: List[int] = [0]
    run_start: Optional[int] = None
    for idx, flag in enumerate(stationary):
        if flag and run_start is None:
            run_start = idx
        elif not flag and run_start is not None:
            centers.append((run_start + idx - 1) // 2)
            run_start = None
    if run_start is not None:
        centers.append((run_start + len(stationary) - 1) // 2)
    if not centers or centers[-1] != len(velocity_norm):
        centers.append(len(velocity_norm))
    return dedupe_keep_order(max(0, min(len(velocity_norm), value)) for value in centers)


def _select_boundary_positions(candidate_positions: List[int], num_stages: int, length: int) -> List[int]:
    if num_stages <= 1:
        return []
    desired_positions = [round(length * idx / num_stages) for idx in range(1, num_stages)]
    selected: List[int] = []
    used = set()
    for desired in desired_positions:
        available = [pos for pos in candidate_positions if 0 < pos < length and pos not in used]
        if available:
            chosen = min(available, key=lambda pos: (abs(pos - desired), pos))
            used.add(chosen)
            selected.append(int(chosen))
        else:
            selected.append(int(desired))
    selected = sorted(selected)
    for idx, value in enumerate(selected):
        lower = idx + 1
        upper = length - (len(selected) - idx - 1)
        selected[idx] = max(lower, min(upper, value))
    return selected


def infer_stage_ranges_from_zarr_episode(
    episode_state: np.ndarray,
    episode_action: np.ndarray,
    num_stages: int,
    velocity_threshold: float,
) -> List[Tuple[int, int]]:
    length = int(len(episode_state))
    if length == 0:
        return []
    if num_stages == 1:
        return [(0, length)]
    velocity_norm = np.linalg.norm(episode_action - episode_state, axis=1)
    threshold = _choose_stationary_threshold(velocity_norm, velocity_threshold)
    stationary_centers = _find_stationary_centers(velocity_norm, threshold)
    boundaries = _select_boundary_positions(stationary_centers, num_stages, length)
    starts = [0] + boundaries
    ends = boundaries + [length]
    return [(start, end) for start, end in zip(starts, ends) if start < end]

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

    explicit_episode_lengths = load_episode_lengths(args.episode_lengths_json) if args.episode_lengths_json else {}
    zarr_episode_lengths = load_episode_lengths_from_zarr(args.dp3_zarr) if args.dp3_zarr else {}
    zarr_episode_state_action = load_episode_state_action_from_zarr(args.dp3_zarr) if args.dp3_zarr and args.boundary_mode == "zarr_velocity" else {}
    episode_lengths = merge_episode_lengths(explicit_episode_lengths, zarr_episode_lengths)
    if not episode_lengths:
        raise ValueError("Provide --episode-lengths-json, --dp3-zarr, or both.")
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
                row = normalize_decomposition_row(row, expand_stack_execution=True)
            episode_length = episode_lengths[episode_id]
            stages = row["decomposition"]["stages"]
            if not stages:
                raise ValueError(f"No stages found for episode {episode_id}.")
            if episode_id in stage_boundaries:
                stage_ranges = normalize_stage_boundaries(stage_boundaries[episode_id], len(stages), episode_length)
            elif args.boundary_mode == "zarr_velocity" and episode_id in zarr_episode_state_action:
                stage_ranges = infer_stage_ranges_from_zarr_episode(
                    episode_state=zarr_episode_state_action[episode_id]["state"],
                    episode_action=zarr_episode_state_action[episode_id]["action"],
                    num_stages=len(stages),
                    velocity_threshold=args.velocity_threshold,
                )
            else:
                stage_ranges = evenly_partition(episode_length, len(stages))
            if len(stage_ranges) != len(stages):
                raise ValueError(
                    f"Episode {episode_id} produced {len(stage_ranges)} ranges for {len(stages)} stages."
                )

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

    expected_total_rows = sum(episode_lengths[episode_id] for episode_id in episode_instruction_map if episode_id in episode_lengths)
    written_episode_ids = sorted(
        episode_id for episode_id in episode_instruction_map
        if episode_id not in {item["episode_id"] for item in missing_instruction_matches}
    )
    per_episode_lengths = {
        str(episode_id): episode_lengths[episode_id]
        for episode_id in written_episode_ids
    }
    flat_index_contiguous = total_rows == flat_index == sum(per_episode_lengths.values())
    zarr_episode_ends = load_episode_ends_from_zarr(args.dp3_zarr) if args.dp3_zarr else None
    zarr_total_steps = zarr_episode_ends[-1] if zarr_episode_ends else None
    zarr_matches_written_rows = zarr_total_steps == total_rows if zarr_total_steps is not None else None

    summary = {
        "decomposition_file": str(Path(args.decomposition_file).resolve()),
        "output": str(output_path.resolve()),
        "episodes_requested": len(episode_instruction_map),
        "episodes_written": episodes_written,
        "rows_written": total_rows,
        "expected_rows_for_requested_episodes": expected_total_rows,
        "missing_instruction_matches": missing_instruction_matches,
        "matched_episode_ids": written_episode_ids,
        "instruction_source": args.instruction_source if args.instruction_dir else "episode_instruction_map",
        "alignment_mode": "explicit_stage_boundaries" if stage_boundaries else args.boundary_mode,
        "stage_normalization": not args.disable_stage_normalization,
        "episode_lengths_source": {
            "json": bool(args.episode_lengths_json),
            "dp3_zarr": bool(args.dp3_zarr),
        },
        "validation": {
            "flat_index_start": 0,
            "flat_index_end_exclusive": flat_index,
            "flat_index_contiguous": flat_index_contiguous,
            "per_episode_lengths": per_episode_lengths,
            "zarr_episode_count": len(zarr_episode_ends) if zarr_episode_ends else None,
            "zarr_total_steps": zarr_total_steps,
            "zarr_matches_written_rows": zarr_matches_written_rows,
        },
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
