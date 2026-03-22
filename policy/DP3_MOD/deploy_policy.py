
# -*- coding: utf-8 -*-
"""
deploy_policy_online_planner_patched.py

Patch goals:
1. stage_name/support_kind/source_object_id are aligned with training sidecar semantics
2. stable object IDs: obj_0/obj_1/obj_2 ; colors are attributes, not IDs
3. no longer use robot_state[:3] as fake EE xyz
4. online planner generates planner_context from current scene + instruction
"""

import sys
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import sapien.core as sapien  # noqa: F401
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None
    print("Warning: scikit-learn is not installed. DBSCAN clustering will be disabled.")

from envs import *  # noqa: F401,F403
from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *  # noqa: F401,F403


def _l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def _as_np3(x, default=None):
    if x is None:
        if default is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(default, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return arr[:3]
    out = np.zeros(3, dtype=np.float32)
    out[:arr.size] = arr
    return out


def _mean_xyz(points: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.mean(points, axis=0).astype(np.float32)


class SceneTracker:
    """
    Tracks tabletop blocks and assigns stable IDs obj_0/obj_1/obj_2.
    """
    def __init__(self):
        self.object_bank: List[Dict[str, Any]] = []
        self.table_z = 0.76
        self.workspace = {
            "x_min": 0.0, "x_max": 0.8,
            "y_min": -0.6, "y_max": 0.6,
            "z_min": self.table_z + 0.015, "z_max": 1.3,
        }
        self.color_refs = {
            "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        self.initialized = False

    def reset(self):
        self.object_bank = []
        self.initialized = False

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float32)
        if np.max(rgb) > 1.5:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)

    def _infer_color(self, mean_rgb: np.ndarray) -> str:
        mean_rgb = self._normalize_rgb(mean_rgb)
        best_name, best_dist = "unknown", 1e9
        for name, ref in self.color_refs.items():
            d = float(np.linalg.norm(mean_rgb - ref))
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

    def _crop_workspace(self, xyz: np.ndarray, rgb: Optional[np.ndarray]):
        w = self.workspace
        mask = (
            (xyz[:, 0] > w["x_min"]) & (xyz[:, 0] < w["x_max"]) &
            (xyz[:, 1] > w["y_min"]) & (xyz[:, 1] < w["y_max"]) &
            (xyz[:, 2] > w["z_min"]) & (xyz[:, 2] < w["z_max"])
        )
        xyz = xyz[mask]
        rgb = rgb[mask] if rgb is not None else None
        return xyz, rgb

    def _remove_robot_points(self, xyz: np.ndarray, rgb: Optional[np.ndarray], runtime_state: Dict[str, Any]):
        ee_left = runtime_state.get("ee_left_xyz") if runtime_state else None
        ee_right = runtime_state.get("ee_right_xyz") if runtime_state else None
        mask = np.ones(len(xyz), dtype=bool)
        for ee in [ee_left, ee_right]:
            if ee is None:
                continue
            ee = _as_np3(ee, default=None)
            if np.allclose(ee, 0.0):
                continue
            d = np.linalg.norm(xyz - ee[None, :], axis=1)
            mask &= (d > 0.08)
        xyz = xyz[mask]
        rgb = rgb[mask] if rgb is not None else None
        return xyz, rgb

    def _cluster(self, xyz: np.ndarray, rgb: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        if DBSCAN is None or len(xyz) < 15:
            return clusters

        labels = DBSCAN(eps=0.03, min_samples=10).fit_predict(xyz)
        for k in sorted(set(labels)):
            if k == -1:
                continue
            cmask = labels == k
            pts = xyz[cmask]
            if len(pts) < 20 or len(pts) > 4000:
                continue
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            size = maxs - mins
            if size[0] > 0.20 or size[1] > 0.20 or size[2] > 0.20:
                continue
            mean_rgb = np.mean(rgb[cmask], axis=0) if rgb is not None else np.zeros(3, dtype=np.float32)
            clusters.append({
                "centroid": _mean_xyz(pts),
                "bbox_min": mins.astype(np.float32),
                "bbox_max": maxs.astype(np.float32),
                "size": size.astype(np.float32),
                "num_points": int(len(pts)),
                "mean_rgb": self._normalize_rgb(mean_rgb),
                "color_name": self._infer_color(mean_rgb),
            })
        return clusters

    def _bootstrap_ids(self, clusters: List[Dict[str, Any]]):
        if not clusters:
            return
        clusters = sorted(clusters, key=lambda c: (float(c["centroid"][0]), float(c["centroid"][1])))
        self.object_bank = []
        for i, c in enumerate(clusters[:3]):
            self.object_bank.append({
                "id": f"obj_{i}",
                "centroid": c["centroid"],
                "bbox_min": c["bbox_min"],
                "bbox_max": c["bbox_max"],
                "size": c["size"],
                "num_points": c["num_points"],
                "color_name": c["color_name"],
                "mean_rgb": c["mean_rgb"],
                "visible": True,
            })
        self.initialized = len(self.object_bank) > 0

    def _match_update(self, clusters: List[Dict[str, Any]]):
        if not self.object_bank:
            self._bootstrap_ids(clusters)
            return
        remaining = clusters.copy()
        new_bank: List[Dict[str, Any]] = []
        for obj in self.object_bank:
            if not remaining:
                new_obj = dict(obj)
                new_obj["visible"] = False
                new_bank.append(new_obj)
                continue
            centroids = np.stack([c["centroid"] for c in remaining], axis=0)
            dists = np.linalg.norm(centroids - obj["centroid"][None, :], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < 0.12:
                c = remaining.pop(j)
                new_bank.append({
                    "id": obj["id"],
                    "centroid": c["centroid"],
                    "bbox_min": c["bbox_min"],
                    "bbox_max": c["bbox_max"],
                    "size": c["size"],
                    "num_points": c["num_points"],
                    "color_name": c["color_name"],
                    "mean_rgb": c["mean_rgb"],
                    "visible": True,
                })
            else:
                new_obj = dict(obj)
                new_obj["visible"] = False
                new_bank.append(new_obj)
        self.object_bank = new_bank

    def update(self, obs: Dict[str, Any], runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        pc = obs.get("point_cloud")
        if pc is None or len(pc) == 0:
            return {"objects": self.object_bank, "table_center": np.array([0.0, 0.0, self.table_z], dtype=np.float32)}
        xyz = pc[:, :3]
        rgb = pc[:, 3:6] if pc.shape[1] >= 6 else None
        xyz, rgb = self._crop_workspace(xyz, rgb)
        xyz, rgb = self._remove_robot_points(xyz, rgb, runtime_state)
        clusters = self._cluster(xyz, rgb)
        if not self.initialized:
            self._bootstrap_ids(clusters)
        else:
            self._match_update(clusters)
        return {
            "objects": self.object_bank,
            "table_center": np.array([0.0, 0.0, self.table_z], dtype=np.float32),
        }


class RulePlanner:
    """
    Online planner aligned with training-side planner semantics.
    stage_name: place_base / stack_middle / stack_top
    support_kind: table_center / stack_anchor
    source_object_id: obj_0 / obj_1 / obj_2
    """
    def __init__(self):
        self.current_stage = None
        self.current_phase = "approach"
        self.block_height = 0.04
        self.target_tower: List[str] = []
        self.color_to_obj: Dict[str, str] = {}
        self.stage_to_arm_default = {
            "place_base": "R",
            "stack_middle": "L",
            "stack_top": "L",
        }

    def reset(self):
        self.current_stage = None
        self.current_phase = "approach"
        self.target_tower = []
        self.color_to_obj = {}

    def _bind_colors(self, scene: Dict[str, Any]):
        self.color_to_obj = {}
        for obj in scene["objects"]:
            cname = obj.get("color_name", "unknown")
            if cname in ("red", "green", "blue") and cname not in self.color_to_obj:
                self.color_to_obj[cname] = obj["id"]

    def _parse_instruction(self, instruction: str) -> List[str]:
        s = instruction.lower().strip()
        if ("blue block on red block" in s and "red block on green block" in s):
            return ["green", "red", "blue"]
        if ("red block on green block" in s and "green block on blue block" in s):
            return ["blue", "green", "red"]
        if ("green block on blue block" in s and "blue block on red block" in s):
            return ["red", "blue", "green"]
        if ("blue block on green block" in s and "green block on red block" in s):
            return ["red", "green", "blue"]
        if "green block on top" in s or "green on top" in s:
            mentioned = []
            for c in ["red", "blue", "green"]:
                if c in s:
                    mentioned.append(c)
            others = [c for c in mentioned if c != "green"]
            if len(others) >= 2:
                return [others[0], others[1], "green"]
        mentioned = []
        for c in ["red", "green", "blue"]:
            if c in s:
                mentioned.append(c)
        if len(mentioned) == 3:
            return mentioned
        return ["red", "green", "blue"]

    def _resolve_target_tower(self, instruction: str, scene: Dict[str, Any]):
        if self.target_tower:
            return
        self._bind_colors(scene)
        color_order = self._parse_instruction(instruction)
        tower = []
        for c in color_order:
            if c in self.color_to_obj:
                tower.append(self.color_to_obj[c])
        if len(tower) < 3:
            by_id = sorted([o["id"] for o in scene["objects"] if str(o["id"]).startswith("obj_")])
            tower = by_id[:3]
        self.target_tower = tower

    def _scene_obj(self, scene: Dict[str, Any], obj_id: str) -> Optional[Dict[str, Any]]:
        for o in scene["objects"]:
            if o["id"] == obj_id:
                return o
        return None

    def _is_near_center(self, centroid: np.ndarray, table_center: np.ndarray, thr=0.06) -> bool:
        return _l2(centroid[:2], table_center[:2]) < thr

    def _is_on_top_of(self, upper: np.ndarray, lower: np.ndarray, xy_thr=0.05, z_gap=0.02) -> bool:
        return (_l2(upper[:2], lower[:2]) < xy_thr) and (upper[2] > lower[2] + z_gap)

    def _estimate_progress(self, scene: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        if len(self.target_tower) < 3:
            return False, False, False
        bottom = self._scene_obj(scene, self.target_tower[0])
        middle = self._scene_obj(scene, self.target_tower[1])
        top = self._scene_obj(scene, self.target_tower[2])
        table_center = scene["table_center"]
        base_done = bool(bottom and self._is_near_center(bottom["centroid"], table_center))
        middle_done = bool(base_done and bottom and middle and self._is_on_top_of(middle["centroid"], bottom["centroid"]))
        top_done = bool(middle_done and middle and top and self._is_on_top_of(top["centroid"], middle["centroid"]))
        return base_done, middle_done, top_done

    def _select_stage(self, scene: Dict[str, Any]):
        table_center = scene["table_center"]
        base_done, middle_done, top_done = self._estimate_progress(scene)
        if len(self.target_tower) < 3:
            return "place_base", "obj_0", "table_center", np.zeros(3, dtype=np.float32), table_center
        bottom_id, middle_id, top_id = self.target_tower
        bottom_obj = self._scene_obj(scene, bottom_id)
        middle_obj = self._scene_obj(scene, middle_id)
        if not base_done:
            src = bottom_id
            source_anchor = _as_np3(bottom_obj["centroid"] if bottom_obj else None)
            support_anchor = _as_np3(table_center)
            return "place_base", src, "table_center", source_anchor, support_anchor
        if not middle_done:
            src = middle_id
            source_anchor = _as_np3(middle_obj["centroid"] if middle_obj else None)
            support_anchor = _as_np3(bottom_obj["centroid"] if bottom_obj else None, default=table_center) + np.array([0,0,self.block_height], dtype=np.float32)
            return "stack_middle", src, "stack_anchor", source_anchor, support_anchor
        if not top_done:
            src = top_id
            top_obj = self._scene_obj(scene, top_id)
            source_anchor = _as_np3(top_obj["centroid"] if top_obj else None)
            support_anchor = _as_np3(middle_obj["centroid"] if middle_obj else None, default=table_center) + np.array([0,0,self.block_height], dtype=np.float32)
            return "stack_top", src, "stack_anchor", source_anchor, support_anchor
        support_anchor = _as_np3(middle_obj["centroid"] if middle_obj else None, default=table_center)
        return "stack_top", top_id, "stack_anchor", support_anchor, support_anchor

    def _select_active_arm(self, stage_name: str, source_anchor: np.ndarray, runtime_state: Dict[str, Any]) -> str:
        ee_left = runtime_state.get("ee_left_xyz")
        ee_right = runtime_state.get("ee_right_xyz")
        if ee_left is not None and ee_right is not None:
            dL = _l2(_as_np3(ee_left), source_anchor)
            dR = _l2(_as_np3(ee_right), source_anchor)
            return "L" if dL < dR else "R"
        return self.stage_to_arm_default.get(stage_name, "R")

    def _gripper_closed(self, active_arm: str, runtime_state: Dict[str, Any]) -> bool:
        return bool(runtime_state.get("gripper_left_closed", False)) if active_arm == "L" else bool(runtime_state.get("gripper_right_closed", False))

    def _ee_pos(self, active_arm: str, runtime_state: Dict[str, Any]) -> np.ndarray:
        return _as_np3(runtime_state.get("ee_left_xyz")) if active_arm == "L" else _as_np3(runtime_state.get("ee_right_xyz"))

    def _update_phase(self, stage_name: str, source_anchor: np.ndarray, support_anchor: np.ndarray, active_arm: str, runtime_state: Dict[str, Any]):
        if self.current_stage != stage_name:
            self.current_stage = stage_name
            self.current_phase = "approach"
        ee = self._ee_pos(active_arm, runtime_state)
        closed = self._gripper_closed(active_arm, runtime_state)
        dist_to_src = _l2(ee, source_anchor)
        dist_to_sup = _l2(ee, support_anchor)
        if self.current_phase == "approach":
            if dist_to_src < 0.06:
                self.current_phase = "close"
        elif self.current_phase == "close":
            if closed:
                self.current_phase = "hold"
        elif self.current_phase == "hold":
            if dist_to_sup < 0.06:
                self.current_phase = "open"
        elif self.current_phase == "open":
            if not closed:
                self.current_phase = "retreat"
        elif self.current_phase == "retreat":
            if dist_to_sup > 0.15:
                pass

    def update(self, instruction: str, scene: Dict[str, Any], runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        self._resolve_target_tower(instruction, scene)
        stage_name, source_id, support_kind, source_anchor, support_anchor = self._select_stage(scene)
        active_arm = self._select_active_arm(stage_name, source_anchor, runtime_state)
        self._update_phase(stage_name, source_anchor, support_anchor, active_arm, runtime_state)
        target_object_id = "center"
        if support_kind == "stack_anchor":
            if stage_name == "stack_middle" and len(self.target_tower) >= 1:
                target_object_id = self.target_tower[0]
            elif stage_name == "stack_top" and len(self.target_tower) >= 2:
                target_object_id = self.target_tower[1]
        return {
            "stage_name": stage_name,
            "phase_name": self.current_phase,
            "active_arm": active_arm,
            "source_object_id": source_id,
            "target_object_id": target_object_id,
            "support_kind": support_kind,
            "source_anchor_xyz": source_anchor.tolist(),
            "support_anchor_xyz": support_anchor.tolist(),
            "planner_weight_source": 1.0,
            "planner_weight_support": 0.35,
            "planner_weight_base": 0.25,
        }


def encode_obs(observation):
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs


def _resolve_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    candidates = [os.path.join(parent_directory, path_value), os.path.abspath(path_value)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path_value)


def _load_jsonl(path_value: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path_value, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_vocab(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    vocab = {'__unk__': 0}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = str(value)
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def _safe_long_tensor(value: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(value)], device=device, dtype=torch.long)


def _safe_float_tensor(values: Optional[List[float]], dims: int, device: torch.device) -> torch.Tensor:
    if values is None:
        values = [0.0] * dims
    arr = np.asarray(values, dtype=np.float32).reshape(1, dims)
    return torch.from_numpy(arr).to(device=device)


class PlannerConditionRuntime:
    def __init__(self, policy, usr_args: Dict[str, Any]):
        self.policy = policy
        self.usr_args = usr_args
        self.step_count = 0
        self.mode = 'none'
        self.debug = bool(usr_args.get('planner_debug', False))
        self.use_planner_condition = bool(getattr(policy, 'use_planner_condition', False))
        if self.use_planner_condition:
            self.mode = 'sidecar'
        self.planner_rows: List[Dict[str, Any]] = []
        self.sidecar_vocab: Dict[str, Dict[str, int]] = {}
        self.default_context: Dict[str, Any] = {}

        planner_labels_jsonl = _resolve_path(usr_args.get('planner_labels_jsonl'))
        if planner_labels_jsonl and os.path.exists(planner_labels_jsonl):
            self.planner_rows = _load_jsonl(planner_labels_jsonl)
            if self.use_planner_condition:
                self.sidecar_vocab = {
                    'stage': _build_vocab(self.planner_rows, 'stage_name'),
                    'phase': _build_vocab(self.planner_rows, 'phase_name'),
                    'arm': _build_vocab(self.planner_rows, 'active_arm'),
                    'source': _build_vocab(self.planner_rows, 'source_object_id'),
                    'support_kind': _build_vocab(self.planner_rows, 'support_kind'),
                }
        if self.debug:
            print(f"[PlannerConditionRuntime] mode={self.mode} rows={len(self.planner_rows)} stage_vocab={len(self.sidecar_vocab.get('stage', {}))} source_vocab={len(self.sidecar_vocab.get('source', {}))}")

    def reset(self):
        self.step_count = 0

    def advance(self, num_steps: int = 1):
        self.step_count += int(num_steps)

    def _lookup(self, vocab: Dict[str, int], value: Any) -> int:
        if value is None:
            return 0
        return int(vocab.get(str(value), 0))

    def current_context(self, task_env=None, observation=None) -> Dict[str, Any]:
        if isinstance(observation, dict) and isinstance(observation.get('planner_context'), dict):
            return observation['planner_context']
        if task_env is not None and hasattr(task_env, 'get_planner_context'):
            try:
                ctx = task_env.get_planner_context()
                if isinstance(ctx, dict):
                    return ctx
            except Exception:
                pass
        return dict(self.default_context)

    def encode_for_policy(self, device: torch.device, task_env=None, observation=None) -> Dict[str, torch.Tensor]:
        if self.mode == 'none':
            return {}
        ctx = self.current_context(task_env=task_env, observation=observation)
        source_anchor = ctx.get('source_anchor_xyz')
        support_anchor = ctx.get('support_anchor_xyz')
        target_anchor = ctx.get('target_anchor_xyz', support_anchor)
        out = {
            'planner_stage_id': _safe_long_tensor(self._lookup(self.sidecar_vocab.get('stage', {}), ctx.get('stage_name')), device),
            'planner_phase_id': _safe_long_tensor(self._lookup(self.sidecar_vocab.get('phase', {}), ctx.get('phase_name')), device),
            'planner_arm_id': _safe_long_tensor(self._lookup(self.sidecar_vocab.get('arm', {}), ctx.get('active_arm')), device),
            'planner_source_id': _safe_long_tensor(self._lookup(self.sidecar_vocab.get('source', {}), ctx.get('source_object_id')), device),
            'planner_support_kind_id': _safe_long_tensor(self._lookup(self.sidecar_vocab.get('support_kind', {}), ctx.get('support_kind')), device),
            'planner_source_anchor': _safe_float_tensor(source_anchor, 3, device),
            'planner_support_anchor': _safe_float_tensor(support_anchor, 3, device),
            'planner_target_anchor': _safe_float_tensor(target_anchor, 3, device),
            'planner_source_valid': _safe_float_tensor([1.0 if source_anchor is not None else 0.0], 1, device),
            'planner_support_valid': _safe_float_tensor([1.0 if support_anchor is not None else 0.0], 1, device),
            'planner_target_valid': _safe_float_tensor([1.0 if target_anchor is not None else 0.0], 1, device),
            'planner_weight_source': _safe_float_tensor([float(ctx.get('planner_weight_source', 1.0))], 1, device),
            'planner_weight_support': _safe_float_tensor([float(ctx.get('planner_weight_support', 0.35))], 1, device),
            'planner_weight_base': _safe_float_tensor([float(ctx.get('planner_weight_base', 0.25))], 1, device),
        }
        if self.debug:
            print("[PlannerContext]", {
                "stage_name": ctx.get("stage_name"),
                "phase_name": ctx.get("phase_name"),
                "active_arm": ctx.get("active_arm"),
                "source_object_id": ctx.get("source_object_id"),
                "support_kind": ctx.get("support_kind"),
                "source_anchor_xyz": ctx.get("source_anchor_xyz"),
                "support_anchor_xyz": ctx.get("support_anchor_xyz"),
            })
        return out


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"
    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)
    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"
    hydra_runtime_cfg = {
        "job": {"override_dirname": usr_args['task_name']},
        "run": {"dir": run_dir},
        "sweep": {"dir": run_dir, "subdir": "0"}
    }
    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args.get('use_rgb', True)

    planner_labels_jsonl = usr_args.get('planner_labels_jsonl')
    if planner_labels_jsonl:
        try:
            cfg.task.dataset.planner_labels_jsonl = planner_labels_jsonl
        except Exception:
            pass
    if 'use_planner_condition' in usr_args:
        try:
            cfg.policy.use_planner_condition = bool(usr_args['use_planner_condition'])
        except Exception:
            pass
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    DP3_Model.planner_runtime = PlannerConditionRuntime(DP3_Model.policy, usr_args)
    return DP3_Model


def _stacked_policy_input(model, task_env=None, observation=None):
    assert len(model.env_runner.obs) > 0, 'no observation is recorded, please update obs first'
    obs = model.env_runner.get_n_steps_obs()
    device = model.policy.device
    obs_input = {
        'point_cloud': torch.from_numpy(obs['point_cloud']).to(device=device).unsqueeze(0),
        'agent_pos': torch.from_numpy(obs['agent_pos']).to(device=device).unsqueeze(0),
    }
    planner_runtime = getattr(model, 'planner_runtime', None)
    if planner_runtime is not None:
        obs_input.update(planner_runtime.encode_for_policy(device=device, task_env=task_env, observation=observation))
    return obs_input


def _predict_actions(model, task_env=None, observation=None):
    with torch.no_grad():
        obs_input = _stacked_policy_input(model, task_env=task_env, observation=observation)
        action_dict = model.policy.predict_action(obs_input)
    action = action_dict['action'].detach().cpu().numpy().squeeze(0)
    return action


def get_runtime_robot_state(task_env, observation, obs_agent_pos) -> Dict[str, Any]:
    """
    Strongly recommended env hook:
        TASK_ENV.get_planner_robot_state()
    returning:
        {
          "ee_left_xyz": [x,y,z],
          "ee_right_xyz": [x,y,z],
          "gripper_left_closed": bool,
          "gripper_right_closed": bool,
        }
    """
    if task_env is not None and hasattr(task_env, 'get_planner_robot_state'):
        try:
            state = task_env.get_planner_robot_state()
            if isinstance(state, dict):
                return state
        except Exception:
            pass
    return {
        "ee_left_xyz": [0.0, 0.0, 0.0],
        "ee_right_xyz": [0.0, 0.0, 0.0],
        "gripper_left_closed": False,
        "gripper_right_closed": False,
    }


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)

    if not hasattr(model, 'scene_tracker'):
        model.scene_tracker = SceneTracker()
        model.rule_planner = RulePlanner()

    raw_instruction = "stack blocks"
    if hasattr(TASK_ENV, 'get_task_text'):
        try:
            raw_instruction = TASK_ENV.get_task_text()
        except Exception:
            pass
    elif hasattr(TASK_ENV, '_active_task_string'):
        raw_instruction = TASK_ENV._active_task_string
        if callable(raw_instruction):
            raw_instruction = raw_instruction()

    runtime_state = get_runtime_robot_state(TASK_ENV, observation, obs['agent_pos'])
    scene = model.scene_tracker.update(obs, runtime_state)
    planner_context = model.rule_planner.update(raw_instruction, scene, runtime_state)
    observation['planner_context'] = planner_context

    if len(model.env_runner.obs) == 0:
        model.update_obs(obs)

    actions = _predict_actions(model, task_env=TASK_ENV, observation=observation)

    for action in actions:
        TASK_ENV.take_action(action)
        planner_runtime = getattr(model, 'planner_runtime', None)
        if planner_runtime is not None:
            planner_runtime.advance(1)

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        runtime_state = get_runtime_robot_state(TASK_ENV, observation, obs['agent_pos'])
        scene = model.scene_tracker.update(obs, runtime_state)
        planner_context = model.rule_planner.update(raw_instruction, scene, runtime_state)
        observation['planner_context'] = planner_context

        model.update_obs(obs)


def reset_model(model):
    model.env_runner.reset_obs()
    planner_runtime = getattr(model, 'planner_runtime', None)
    if planner_runtime is not None:
        planner_runtime.reset()
    if hasattr(model, 'scene_tracker'):
        model.scene_tracker.reset()
    if hasattr(model, 'rule_planner'):
        model.rule_planner.reset()
