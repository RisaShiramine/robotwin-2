import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict, Optional, Tuple
import json
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def resolve_repo_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(parent_directory, path)


def _episode_id_and_timestep_arrays(episode_ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    episode_ids = np.zeros(total, dtype=np.int64)
    timesteps = np.zeros(total, dtype=np.int64)
    start = 0
    for episode_id, end in enumerate(np.asarray(episode_ends, dtype=np.int64)):
        end = int(end)
        episode_ids[start:end] = episode_id
        timesteps[start:end] = np.arange(end - start, dtype=np.int64)
        start = end
    return episode_ids, timesteps


def load_and_validate_planner_labels(planner_labels_jsonl: str, replay_buffer: ReplayBuffer):
    planner_labels_jsonl = resolve_repo_path(planner_labels_jsonl)
    with open(planner_labels_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda row: int(row["flat_index"]))

    expected = int(replay_buffer["action"].shape[0])
    if len(rows) != expected:
        raise ValueError(f"Planner label count {len(rows)} does not match replay buffer length {expected}.")

    expected_flat_indices = np.arange(expected, dtype=np.int64)
    actual_flat_indices = np.asarray([int(row["flat_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(actual_flat_indices, expected_flat_indices):
        raise ValueError("Planner labels must contain contiguous flat_index values matching replay buffer transitions.")

    episode_ids, timesteps = _episode_id_and_timestep_arrays(np.asarray(replay_buffer.episode_ends[:], dtype=np.int64))
    actual_episode_ids = np.asarray([int(row["episode_id"]) for row in rows], dtype=np.int64)
    timestep_key = "timestep" if "timestep" in rows[0] else "episode_timestep"
    actual_timesteps = np.asarray([int(row[timestep_key]) for row in rows], dtype=np.int64)
    if not np.array_equal(actual_episode_ids, episode_ids):
        raise ValueError("Planner label episode_id values do not align with replay buffer episode boundaries.")
    if not np.array_equal(actual_timesteps, timesteps):
        raise ValueError("Planner label timestep values do not align with replay buffer episode-relative timesteps.")
    return rows


def build_vocab(rows, key: str):
    vocab = {"__unk__": 0}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def inspect_planner_sidecar(planner_labels_jsonl: Optional[str]) -> Tuple[bool, Dict[str, int]]:
    if not planner_labels_jsonl:
        return False, {}
    planner_labels_jsonl = resolve_repo_path(planner_labels_jsonl)
    if not os.path.exists(planner_labels_jsonl):
        raise FileNotFoundError(f"planner_labels_jsonl not found: {planner_labels_jsonl}")
    with open(planner_labels_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"planner_labels_jsonl is empty: {planner_labels_jsonl}")
    vocab_sizes = {
        "stage": len(build_vocab(rows, "stage_name")),
        "phase": len(build_vocab(rows, "phase_name")),
        "arm": len(build_vocab(rows, "active_arm")),
        "source": len(build_vocab(rows, "source_object_id")),
        "support_kind": len(build_vocab(rows, "support_kind")),
    }
    return True, vocab_sizes


class RobotDataset(BaseDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        planner_labels_jsonl=None,
        n_obs_steps=1,
    ):
        super().__init__()
        self.task_name = task_name
        self.n_obs_steps = int(n_obs_steps)
        zarr_path = resolve_repo_path(zarr_path)
        replay_root = ReplayBuffer.create_from_path(zarr_path)
        keys = ["state", "action", "point_cloud"]
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        self.has_planner_sidecar = planner_labels_jsonl is not None
        self.planner_labels = None
        self.planner_vocab = {}
        if self.has_planner_sidecar:
            self.planner_labels = load_and_validate_planner_labels(planner_labels_jsonl, self.replay_buffer)
            self.planner_vocab = {
                "stage": build_vocab(self.planner_labels, "stage_name"),
                "phase": build_vocab(self.planner_labels, "phase_name"),
                "arm": build_vocab(self.planner_labels, "active_arm"),
                "source": build_vocab(self.planner_labels, "source_object_id"),
                "support_kind": build_vocab(self.planner_labels, "support_kind"),
            }

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        point_cloud = sample["point_cloud"].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        return data

    def _planner_index_for_sample(self, idx: int) -> int:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        obs_slot = self.n_obs_steps - 1
        if obs_slot < sample_start_idx:
            return int(buffer_start_idx)
        rel = obs_slot - int(sample_start_idx)
        rel = min(rel, int(buffer_end_idx - buffer_start_idx - 1))
        rel = max(rel, 0)
        return int(buffer_start_idx + rel)

    def _encode_planner_row(self, row: Dict):
        weight_cfg = row.get("recommended_weights", {}) or {}
        source_anchor = row.get("source_anchor_xyz")
        support_anchor = row.get("support_anchor_xyz")
        target_anchor = row.get("target_anchor_xyz")
        source_valid = source_anchor is not None
        support_valid = support_anchor is not None
        target_valid = target_anchor is not None
        return {
            "planner_stage_id": np.array(self.planner_vocab["stage"].get(row.get("stage_name"), 0), dtype=np.int64),
            "planner_phase_id": np.array(self.planner_vocab["phase"].get(row.get("phase_name"), 0), dtype=np.int64),
            "planner_arm_id": np.array(self.planner_vocab["arm"].get(row.get("active_arm"), 0), dtype=np.int64),
            "planner_source_id": np.array(self.planner_vocab["source"].get(row.get("source_object_id"), 0), dtype=np.int64),
            "planner_support_kind_id": np.array(
                self.planner_vocab["support_kind"].get(row.get("support_kind"), 0), dtype=np.int64
            ),
            "planner_source_anchor": np.asarray(source_anchor if source_anchor is not None else [0.0, 0.0, 0.0], dtype=np.float32),
            "planner_support_anchor": np.asarray(support_anchor if support_anchor is not None else [0.0, 0.0, 0.0], dtype=np.float32),
            "planner_target_anchor": np.asarray(target_anchor if target_anchor is not None else [0.0, 0.0, 0.0], dtype=np.float32),
            "planner_source_valid": np.array(float(source_valid), dtype=np.float32),
            "planner_support_valid": np.array(float(support_valid), dtype=np.float32),
            "planner_target_valid": np.array(float(target_valid), dtype=np.float32),
            "planner_weight_source": np.array(float(weight_cfg.get("source", 1.0)), dtype=np.float32),
            "planner_weight_support": np.array(float(weight_cfg.get("support", 0.35)), dtype=np.float32),
            "planner_weight_base": np.array(float(weight_cfg.get("other_objects", 0.25)), dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        if self.has_planner_sidecar:
            planner_row = self.planner_labels[self._planner_index_for_sample(idx)]
            data.update(self._encode_planner_row(planner_row))
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
