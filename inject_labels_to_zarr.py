import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def inject_zarr(zarr_path, jsonl_path, vocab_output=None):
    zarr_path = Path(zarr_path)
    jsonl_path = Path(jsonl_path)
    vocab_output = Path(vocab_output) if vocab_output else zarr_path.parent / "planner_vocab.json"

    with jsonl_path.open("r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]

    labels.sort(key=lambda x: x["flat_index"])

    stage_vocab = {"unknown": 0}
    obj_vocab = {"unknown": 0, "null": 1}

    def get_stage_id(val):
        if val is None:
            return stage_vocab["unknown"]
        val = str(val).lower().strip()
        if val not in stage_vocab:
            stage_vocab[val] = len(stage_vocab)
        return stage_vocab[val]

    def get_obj_id(val):
        if val is None:
            return obj_vocab["null"]
        val = str(val).lower().strip()
        if not val:
            return obj_vocab["null"]
        if val not in obj_vocab:
            obj_vocab[val] = len(obj_vocab)
        return obj_vocab[val]

    root = zarr.open(str(zarr_path), mode="a")
    data_group = root.require_group("data")
    zarr_frames = int(data_group["action"].shape[0])

    n_frames = len(labels)
    if n_frames != zarr_frames:
        raise ValueError(f"planner_labels.jsonl row count ({n_frames}) must match zarr transition count ({zarr_frames}).")

    expected_indices = np.arange(n_frames, dtype=np.int64)
    actual_indices = np.asarray([int(row["flat_index"]) for row in labels], dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("planner_labels.jsonl must contain contiguous flat_index values starting from 0.")

    stage_ids = np.zeros(n_frames, dtype=np.int64)
    source_ids = np.zeros(n_frames, dtype=np.int64)
    target_ids = np.zeros(n_frames, dtype=np.int64)

    for row in labels:
        idx = int(row["flat_index"])
        stage_ids[idx] = get_stage_id(row.get("stage_type"))
        source_ids[idx] = get_obj_id(row.get("source_object"))
        tgt = row.get("target_object") or row.get("target_region") or row.get("target_support")
        target_ids[idx] = get_obj_id(tgt)

    for name, arr in [("stage_id", stage_ids), ("source_id", source_ids), ("target_id", target_ids)]:
        if name in data_group:
            data_group[name][:] = arr
        else:
            data_group.array(name=name, data=arr, chunks=arr.shape, dtype=np.int64, overwrite=False)

    payload = {
        "stage": stage_vocab,
        "object": obj_vocab,
        "counts": {
            "num_stage_ids": len(stage_vocab),
            "num_object_ids": len(obj_vocab),
        },
    }
    with vocab_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ 成功将 Planner IDs 注入 Zarr: {zarr_path}")
    print(f"📌 总帧数: {n_frames}")
    print(f"📌 planner_vocab.json: {vocab_output}")
    print(f"📌 Stage vocab size: {len(stage_vocab)}")
    print(f"📌 Object vocab size: {len(obj_vocab)}")


def main():
    parser = argparse.ArgumentParser(description="Inject planner token IDs into an existing RoboTwin DP3 zarr dataset.")
    parser.add_argument("--zarr_path", required=True, help="Path to the target .zarr dataset")
    parser.add_argument("--jsonl_path", required=True, help="Path to planner_labels.jsonl")
    parser.add_argument(
        "--vocab_output",
        default=None,
        help="Optional output path for planner_vocab.json. Defaults to the dataset parent directory.",
    )
    args = parser.parse_args()
    inject_zarr(args.zarr_path, args.jsonl_path, args.vocab_output)


if __name__ == "__main__":
    main()
