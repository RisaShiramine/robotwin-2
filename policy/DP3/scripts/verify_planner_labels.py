#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify planner_labels.jsonl against a DP3 zarr and optional episode instruction map.")
    parser.add_argument("--planner-labels", required=True, help="Path to planner_labels.jsonl.")
    parser.add_argument("--dp3-zarr", required=True, help="Path to the DP3 zarr used for training.")
    parser.add_argument("--episode-instruction-map", default=None, help="Optional JSON mapping episode ids to expected instructions.")
    parser.add_argument("--summary-output", default=None, help="Optional path to write the validation summary JSON.")
    return parser.parse_args()


def load_rows(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda row: int(row["flat_index"]))
    return rows


def load_episode_instruction_map(path: Optional[str]) -> Optional[Dict[int, str]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: Dict[int, str] = {}
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            mapping[idx] = value if isinstance(value, str) else value.get("instruction")
        return mapping
    for key, value in payload.items():
        mapping[int(key)] = value if isinstance(value, str) else value.get("instruction")
    return mapping


def load_zarr_episode_ends(zarr_path: str) -> List[int]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zarr is required to verify planner label alignment.") from exc
    root = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    return [int(value) for value in root["meta"]["episode_ends"][:]]


def build_expected_index(episode_ends: List[int]) -> List[Dict[str, int]]:
    expected: List[Dict[str, int]] = []
    start = 0
    for episode_id, end in enumerate(episode_ends):
        for timestep in range(end - start):
            expected.append({"flat_index": start + timestep, "episode_id": episode_id, "timestep": timestep})
        start = end
    return expected


def main() -> None:
    args = parse_args()
    rows = load_rows(args.planner_labels)
    episode_ends = load_zarr_episode_ends(args.dp3_zarr)
    expected_index = build_expected_index(episode_ends)
    expected_instructions = load_episode_instruction_map(args.episode_instruction_map)

    errors: List[str] = []
    warnings: List[str] = []
    if len(rows) != len(expected_index):
        errors.append(
            f"planner_labels row count {len(rows)} does not match zarr transition count {len(expected_index)}"
        )

    compared = min(len(rows), len(expected_index))
    for idx in range(compared):
        row = rows[idx]
        expected = expected_index[idx]
        for key in ("flat_index", "episode_id", "timestep"):
            if int(row[key]) != expected[key]:
                errors.append(
                    f"row {idx} has {key}={row[key]} but expected {expected[key]}"
                )
                break

    instruction_mismatches: List[Dict[str, Any]] = []
    if expected_instructions is not None:
        seen_episode_ids = set()
        for row in rows:
            episode_id = int(row["episode_id"])
            if episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            expected_instruction = expected_instructions.get(episode_id)
            actual_instruction = row.get("instruction")
            if expected_instruction is None:
                warnings.append(f"episode {episode_id} is present in planner_labels but missing from episode_instruction_map")
            elif actual_instruction != expected_instruction:
                instruction_mismatches.append(
                    {
                        "episode_id": episode_id,
                        "expected_instruction": expected_instruction,
                        "actual_instruction": actual_instruction,
                    }
                )

        missing_episodes = sorted(set(expected_instructions) - {int(row["episode_id"]) for row in rows})
        if missing_episodes:
            warnings.append(f"episodes missing from planner_labels: {missing_episodes}")
    else:
        missing_episodes = []

    if instruction_mismatches:
        errors.append(f"found {len(instruction_mismatches)} episode instruction mismatches")

    summary = {
        "planner_labels": str(Path(args.planner_labels).resolve()),
        "dp3_zarr": str(Path(args.dp3_zarr).resolve()),
        "row_count": len(rows),
        "zarr_transition_count": len(expected_index),
        "zarr_episode_count": len(episode_ends),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "instruction_mismatches": instruction_mismatches,
        "missing_episodes": missing_episodes,
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
