import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict, Tuple
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
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb


def resolve_zarr_path(zarr_path: str) -> str:
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    return os.path.join(parent_directory, zarr_path)


def inspect_planner_tokens(zarr_path: str) -> Tuple[bool, Dict[str, int]]:
    resolved_path = resolve_zarr_path(zarr_path)
    replay_root = ReplayBuffer.create_from_path(resolved_path)
    has_tokens = all(name in replay_root.keys() for name in ("stage_id", "source_id", "target_id"))
    vocab_sizes = {}
    if has_tokens:
        vocab_sizes = {
            "stage": int(np.max(replay_root["stage_id"])) + 1 if len(replay_root["stage_id"]) > 0 else 1,
            "object": int(max(np.max(replay_root["source_id"]), np.max(replay_root["target_id"]))) + 1
            if len(replay_root["source_id"]) > 0
            else 1,
        }
    return has_tokens, vocab_sizes


def filter_indices_by_stage(indices: np.ndarray, replay_buffer: ReplayBuffer) -> np.ndarray:
    if len(indices) == 0:
        return indices

    stage_ids = replay_buffer["stage_id"][:]
    keep = np.ones(len(indices), dtype=bool)
    sequence_length = int(indices[0][3]) if len(indices) > 0 else 0
    for i, (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx) in enumerate(indices):
        if sample_start_idx != 0 or sample_end_idx != sequence_length:
            keep[i] = False
            continue
        window_stage_ids = stage_ids[buffer_start_idx:buffer_end_idx]
        if len(window_stage_ids) == 0 or np.any(window_stage_ids != window_stage_ids[0]):
            keep[i] = False
    return indices[keep]


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
    ):
        super().__init__()
        self.task_name = task_name
        zarr_path = resolve_zarr_path(zarr_path)
        replay_root = ReplayBuffer.create_from_path(zarr_path)
        keys = ["state", "action", "point_cloud"]
        self.has_planner_tokens = all(name in replay_root.keys() for name in ("stage_id", "source_id", "target_id"))
        if self.has_planner_tokens:
            keys.extend(["stage_id", "source_id", "target_id"])
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)  # 'img'
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
        if self.has_planner_tokens:
            self.sampler.indices = filter_indices_by_stage(self.sampler.indices, self.replay_buffer)
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
        if self.has_planner_tokens:
            val_set.sampler.indices = filter_indices_by_stage(val_set.sampler.indices, self.replay_buffer)
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
        agent_pos = sample["state"][
            :,
        ].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)  # (T, 1024, 6)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, 1024, 6
                "agent_pos": agent_pos,  # T, D_pos
            },
            "action": sample["action"].astype(np.float32),  # T, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        if self.has_planner_tokens:
            buffer_start_idx = int(self.sampler.indices[idx][0])
            data["stage_id"] = np.array(self.replay_buffer["stage_id"][buffer_start_idx], dtype=np.int64)
            data["source_id"] = np.array(self.replay_buffer["source_id"][buffer_start_idx], dtype=np.int64)
            data["target_id"] = np.array(self.replay_buffer["target_id"][buffer_start_idx], dtype=np.int64)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
