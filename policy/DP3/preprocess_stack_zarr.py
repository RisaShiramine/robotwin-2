#!/usr/bin/env python3
"""
Preprocess a raw zarr-style robot dataset (stored as directory/file paths) into
stage-aware labels for 3-block stacking.

Inputs
------
- point_cloud path : zarr array directory with shape [T, N, 6] (xyzrgb)
- state path       : zarr array directory with shape [T, 14]
- action path      : zarr array directory with shape [T, 14] (optional; only used for consistency checks)
- episode_ends     : raw blosc-compressed chunk containing cumulative episode end indices (int64)

Outputs
-------
- manifest.json
- episode_summary.jsonl
- object_bank.jsonl
- cycle_summary.jsonl
- timestep_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import blosc2
import numpy as np
from sklearn.cluster import DBSCAN


class ZarrPathArray:
    """Minimal zarr array reader for local directory-backed arrays."""

    def __init__(self, root_path: str, max_cache_chunks: int = 2):
        self.root = pathlib.Path(root_path)
        if not self.root.exists():
            raise FileNotFoundError(f"zarr path not found: {self.root}")
        meta_path = self.root / ".zarray"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing .zarray in {self.root}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.shape = tuple(self.meta["shape"])
        self.chunks = tuple(self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.ndim = len(self.shape)
        self.max_cache_chunks = max_cache_chunks
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _chunk_path(self, chunk_idx: int) -> pathlib.Path:
        suffix = ".".join([str(chunk_idx)] + ["0"] * (self.ndim - 1))
        return self.root / suffix

    def _load_chunk(self, chunk_idx: int) -> np.ndarray:
        raw = self._chunk_path(chunk_idx).read_bytes()
        decomp = blosc2.decompress(raw)
        arr = np.frombuffer(decomp, dtype=self.dtype).reshape(self.chunks)
        return arr

    def _get_chunk(self, chunk_idx: int) -> np.ndarray:
        if chunk_idx in self._cache:
            arr = self._cache.pop(chunk_idx)
            self._cache[chunk_idx] = arr
            return arr
        arr = self._load_chunk(chunk_idx)
        self._cache[chunk_idx] = arr
        while len(self._cache) > self.max_cache_chunks:
            self._cache.popitem(last=False)
        return arr

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.shape[0]:
            raise IndexError(idx)
        c0 = idx // self.chunks[0]
        local0 = idx % self.chunks[0]
        chunk = self._get_chunk(c0)
        return np.array(chunk[local0], copy=True)

    def load_all(self) -> np.ndarray:
        out = np.empty(self.shape, dtype=self.dtype)
        for c0 in range(math.ceil(self.shape[0] / self.chunks[0])):
            chunk = self._get_chunk(c0)
            start = c0 * self.chunks[0]
            end = min((c0 + 1) * self.chunks[0], self.shape[0])
            out[start:end] = chunk[: end - start]
        return out


def load_episode_ends(path: str) -> np.ndarray:
    raw = pathlib.Path(path).read_bytes()
    decomp = blosc2.decompress(raw)
    return np.frombuffer(decomp, dtype=np.int64)


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif (not m) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@dataclass
class CycleEvent:
    cycle_idx: int
    active_arm: str
    close_start: int
    close_end: int
    open_start: int
    open_end: int
    approach_start: int
    hold_start: int
    retreat_end: int


ARM_MAP = {"L": 6, "R": 13}


def detect_cycles(state_ep: np.ndarray, close_thresh: float = 0.03, open_thresh: float = 0.03) -> List[CycleEvent]:
    arm_events = []
    for arm_name, ch in ARM_MAP.items():
        g = state_ep[:, ch]
        d = np.diff(g)
        close_runs = contiguous_runs(d < -close_thresh)
        open_runs = contiguous_runs(d > open_thresh)
        for cs, ce in close_runs:
            close_start = cs
            close_end = ce + 1
            open_match = None
            for os, oe in open_runs:
                open_start = os
                open_end = oe + 1
                if open_start >= close_end:
                    open_match = (open_start, open_end)
                    break
            if open_match is None:
                continue
            arm_events.append((close_start, arm_name, close_start, close_end, open_match[0], open_match[1]))
    arm_events.sort(key=lambda x: x[0])
    cleaned = []
    for item in arm_events:
        if cleaned and item[1] == cleaned[-1][1] and abs(item[2] - cleaned[-1][2]) <= 2:
            continue
        cleaned.append(item)
    cleaned = cleaned[:3]
    cycles: List[CycleEvent] = []
    prev_open_end = 0
    for i, (_, arm_name, close_start, close_end, open_start, open_end) in enumerate(cleaned):
        retreat_end = cleaned[i + 1][2] if i + 1 < len(cleaned) else len(state_ep)
        cycles.append(CycleEvent(i, arm_name, close_start, close_end, open_start, open_end, prev_open_end, close_end, retreat_end))
        prev_open_end = open_end
    return cycles


@dataclass
class ObjectProposal:
    object_id: str
    centroid: List[float]
    bbox_min: List[float]
    bbox_max: List[float]
    size: int


@dataclass
class InitialObjectBank:
    frame_index: int
    table_z: float
    table_center_xy: List[float]
    block_height_est: float
    objects: List[ObjectProposal]
    nuisance_anchor: Optional[List[float]]


@dataclass
class CycleSummary:
    episode_id: int
    cycle_idx: int
    active_arm: str
    stage_name: str
    support_kind: str
    source_object_id: Optional[str]
    support_object_id: Optional[str]
    pre_frame: int
    post_frame: int
    source_pre: Optional[List[float]]
    source_post: Optional[List[float]]
    support_centroid: Optional[List[float]]
    target_anchor: Optional[List[float]]


def segment_clusters(points_xyz: np.ndarray, plane_margin: float = 0.008, dbscan_eps: float = 0.018, dbscan_min_samples: int = 12, min_cluster_points: int = 12):
    pts = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if len(pts) == 0:
        return 0.0, []
    table_z = float(np.quantile(pts[:, 2], 0.20))
    fg = pts[pts[:, 2] > table_z + plane_margin]
    if len(fg) == 0:
        return table_z, []
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(fg)
    clusters = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        p = fg[labels == lab]
        if len(p) < min_cluster_points:
            continue
        centroid = p.mean(axis=0)
        bbox_min = p.min(axis=0)
        bbox_max = p.max(axis=0)
        clusters.append({"centroid": centroid, "bbox_min": bbox_min, "bbox_max": bbox_max, "size": int(len(p))})
    return table_z, clusters


def build_initial_object_bank(pc_reader: ZarrPathArray, ep_start: int, static_window: int = 8) -> InitialObjectBank:
    centers = []
    nuisance_anchor = None
    table_zs = []
    cluster_records = []
    for t in range(ep_start, ep_start + static_window):
        pc = pc_reader.get(t)[:, :3]
        table_z, clusters = segment_clusters(pc)
        table_zs.append(table_z)
        cluster_records.extend(clusters)
    stable = [c for c in cluster_records if 0.73 <= c["centroid"][2] <= 0.83]
    stable.sort(key=lambda c: (c["centroid"][2], c["size"]))
    table_like = [c for c in stable if c["centroid"][2] < 0.80]
    high_like = [c for c in stable if c["centroid"][2] >= 0.80]
    table_like.sort(key=lambda c: c["size"], reverse=True)
    chosen = table_like[:3]
    chosen.sort(key=lambda c: c["centroid"][0])
    objects = []
    for i, c in enumerate(chosen):
        objects.append(ObjectProposal(f"obj_{i}", c["centroid"].tolist(), c["bbox_min"].tolist(), c["bbox_max"].tolist(), c["size"]))
        centers.append(c["centroid"])
    if high_like:
        nuisance_anchor = np.mean([c["centroid"] for c in high_like], axis=0).tolist()
    table_center_xy = np.mean(np.stack(centers)[:, :2], axis=0).tolist() if centers else [0.0, 0.0]
    block_height_est = float(np.median([o.bbox_max[2] - o.bbox_min[2] for o in objects])) if objects else 0.04
    return InitialObjectBank(ep_start, float(np.median(table_zs)), table_center_xy, block_height_est, objects, nuisance_anchor)


def stable_window_centroid(pc_reader: ZarrPathArray, frame_idx: int, center_hint: Optional[np.ndarray], radius: float = 0.05) -> Optional[np.ndarray]:
    if center_hint is None:
        return None
    pts = pc_reader.get(frame_idx)[:, :3]
    d = np.linalg.norm(pts - center_hint[None, :], axis=1)
    local = pts[d < radius]
    if len(local) < 10:
        return None
    return np.median(local, axis=0)


def infer_cycle_sources_and_supports(pc_reader: ZarrPathArray, ep_start: int, cycles: List[CycleEvent], bank: InitialObjectBank) -> List[CycleSummary]:
    remaining = [o.object_id for o in bank.objects]
    obj_centers = {o.object_id: np.array(o.centroid, dtype=float) for o in bank.objects}
    center_anchor = np.array([bank.table_center_xy[0], bank.table_center_xy[1], bank.table_z + max(bank.block_height_est, 0.03)], dtype=float)
    summaries: List[CycleSummary] = []
    prev_source = None
    prev_post = None
    for i, cyc in enumerate(cycles):
        stage_name = ["place_base", "stack_middle", "stack_top"][i] if i < 3 else f"cycle_{i}"
        pre_frame = ep_start + max(cyc.close_start - 3, 0)
        post_frame = ep_start + min(cyc.open_end + 3, cyc.retreat_end - 1)
        best_id = None
        best_pre = None
        best_post = None
        best_shift = -1.0
        for oid in list(remaining):
            c0 = obj_centers[oid]
            pre_c = stable_window_centroid(pc_reader, pre_frame, c0)
            post_c = stable_window_centroid(pc_reader, post_frame, c0)
            if pre_c is None:
                pre_c = c0.copy()
            if post_c is None:
                post_c = c0.copy()
            shift = float(np.linalg.norm(post_c - pre_c))
            if shift > best_shift:
                best_shift = shift
                best_id = oid
                best_pre = pre_c
                best_post = post_c
        if best_id is None and remaining:
            best_id = remaining[0]
            best_pre = obj_centers[best_id].copy()
            best_post = best_pre.copy()
        if best_id in remaining:
            remaining.remove(best_id)
        if best_id is not None and best_post is not None:
            obj_centers[best_id] = best_post
        if i == 0:
            support_kind = "table_center"
            support_object_id = None
            support_centroid = center_anchor.tolist()
            target_anchor = center_anchor.tolist()
        else:
            support_kind = "stack_anchor"
            support_object_id = prev_source
            support_centroid = prev_post.tolist() if prev_post is not None else center_anchor.tolist()
            target_anchor = support_centroid
        summaries.append(CycleSummary(-1, i, cyc.active_arm, stage_name, support_kind, best_id, support_object_id, pre_frame, post_frame,
                                      None if best_pre is None else best_pre.tolist(),
                                      None if best_post is None else best_post.tolist(),
                                      support_centroid, target_anchor))
        prev_source = best_id
        prev_post = best_post
    return summaries


def phase_segments_for_cycle(c: CycleEvent, ep_len: int):
    segs = []
    cuts = [
        ("approach", c.approach_start, c.close_start),
        ("close", c.close_start, c.close_end),
        ("hold", c.close_end, c.open_start),
        ("open", c.open_start, c.open_end),
        ("retreat", c.open_end, c.retreat_end),
    ]
    for name, s, e in cuts:
        s = max(0, min(ep_len, s))
        e = max(0, min(ep_len, e))
        if e > s:
            segs.append((name, s, e))
    return segs


def recommended_weights(phase: str, support_exists: bool):
    if phase == "approach":
        return {"source": 1.0, "support": 0.35 if support_exists else 0.15, "other_objects": 0.25, "nuisance": 0.10}
    if phase == "close":
        return {"source": 1.0, "support": 0.25 if support_exists else 0.15, "other_objects": 0.20, "nuisance": 0.10}
    if phase == "hold":
        return {"source": 1.0, "support": 0.45 if support_exists else 0.15, "other_objects": 0.25, "nuisance": 0.10}
    if phase == "open":
        return {"source": 0.75, "support": 0.85 if support_exists else 0.25, "other_objects": 0.20, "nuisance": 0.10}
    if phase == "retreat":
        return {"source": 0.45, "support": 0.65 if support_exists else 0.25, "other_objects": 0.20, "nuisance": 0.10}
    return {"source": 0.4, "support": 0.4, "other_objects": 0.2, "nuisance": 0.1}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-cloud-path", required=True, help="Path to point_cloud zarr array directory")
    ap.add_argument("--state-path", required=True, help="Path to state zarr array directory")
    ap.add_argument("--action-path", required=False, help="Path to action zarr array directory")
    ap.add_argument("--episode-ends", required=True, help="Path to raw blosc episode_ends chunk")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_reader = ZarrPathArray(args.point_cloud_path, max_cache_chunks=2)
    state_reader = ZarrPathArray(args.state_path, max_cache_chunks=4)
    action_reader = ZarrPathArray(args.action_path, max_cache_chunks=4) if args.action_path else None

    state = state_reader.load_all()
    action = action_reader.load_all() if action_reader else None
    episode_ends = load_episode_ends(args.episode_ends)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    manifest = {
        "num_episodes": int(len(episode_ends)),
        "num_steps": int(state.shape[0]),
        "state_shape": list(state.shape),
        "point_cloud_shape": list(pc_reader.shape),
        "episode_end_count": int(len(episode_ends)),
        "gripper_channels": {"left": 6, "right": 13},
        "task_assumption": "3-block stacking",
        "outputs": ["episode_summary.jsonl", "object_bank.jsonl", "cycle_summary.jsonl", "timestep_labels.jsonl"],
        "input_mode": "directory_paths",
    }
    if action is not None:
        manifest["action_equals_next_state_globally_excluding_last_step"] = bool(np.allclose(action[:-1], state[1:], atol=1e-8))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with (out_dir / "episode_summary.jsonl").open("w", encoding="utf-8") as ep_f, \
         (out_dir / "object_bank.jsonl").open("w", encoding="utf-8") as obj_f, \
         (out_dir / "cycle_summary.jsonl").open("w", encoding="utf-8") as cyc_f, \
         (out_dir / "timestep_labels.jsonl").open("w", encoding="utf-8") as step_f:
        for ep_id, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends)):
            ep_start_i, ep_end_i = int(ep_start), int(ep_end)
            state_ep = state[ep_start_i:ep_end_i]
            cycles = detect_cycles(state_ep)
            bank = build_initial_object_bank(pc_reader, ep_start_i)
            cycle_summaries = infer_cycle_sources_and_supports(pc_reader, ep_start_i, cycles, bank)
            for c in cycle_summaries:
                c.episode_id = ep_id

            ep_record = {
                "episode_id": ep_id,
                "global_start": ep_start_i,
                "global_end": ep_end_i,
                "length": ep_end_i - ep_start_i,
                "detected_cycles": len(cycles),
                "active_arm_pattern": "".join(c.active_arm for c in cycles),
                "initial_object_frame": bank.frame_index,
                "table_z": bank.table_z,
                "table_center_xy": bank.table_center_xy,
                "block_height_est": bank.block_height_est,
            }
            ep_f.write(json.dumps(ep_record, ensure_ascii=False) + "\n")

            obj_record = {
                "episode_id": ep_id,
                "frame_index": bank.frame_index,
                "table_z": bank.table_z,
                "table_center_xy": bank.table_center_xy,
                "block_height_est": bank.block_height_est,
                "initial_objects": [o.__dict__ for o in bank.objects],
                "nuisance_anchor": bank.nuisance_anchor,
                "cycles": [c.__dict__ for c in cycle_summaries],
            }
            obj_f.write(json.dumps(obj_record, ensure_ascii=False) + "\n")

            used_frames = set()
            for cyc, cyc_sum in zip(cycles, cycle_summaries):
                cyc_f.write(json.dumps(cyc_sum.__dict__, ensure_ascii=False) + "\n")
                support_exists = cyc_sum.support_centroid is not None
                for phase_name, s, e in phase_segments_for_cycle(cyc, len(state_ep)):
                    for local_t in range(s, e):
                        flat_index = ep_start_i + local_t
                        if flat_index in used_frames:
                            continue
                        used_frames.add(flat_index)
                        phase_id = ["approach", "close", "hold", "open", "retreat"].index(phase_name)
                        row = {
                            "flat_index": flat_index,
                            "episode_id": ep_id,
                            "timestep": local_t,
                            "cycle_idx": cyc.cycle_idx,
                            "stage_name": cyc_sum.stage_name,
                            "phase_name": phase_name,
                            "phase_id": phase_id,
                            "active_arm": cyc.active_arm,
                            "stage_type": "establish_base" if cyc.cycle_idx == 0 else "stack_on_support",
                            "source_object_id": cyc_sum.source_object_id,
                            "support_object_id": cyc_sum.support_object_id,
                            "support_kind": cyc_sum.support_kind,
                            "source_anchor_xyz": cyc_sum.source_pre,
                            "support_anchor_xyz": cyc_sum.support_centroid,
                            "target_anchor_xyz": cyc_sum.target_anchor,
                            "recommended_weights": recommended_weights(phase_name, support_exists),
                        }
                        step_f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
