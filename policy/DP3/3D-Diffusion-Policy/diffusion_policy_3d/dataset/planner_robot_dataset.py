import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import json
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class PlannerRobotDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        planner_labels_jsonl,
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
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        planner_labels_jsonl = os.path.join(parent_directory, planner_labels_jsonl)

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["state", "action", "point_cloud"])
        self.planner_labels = self._load_planner_labels(planner_labels_jsonl)
        self.stage_vocab = self._build_vocab(self.planner_labels, "stage")
        self.stage_type_vocab = self._build_vocab(self.planner_labels, "stage_type")
        self.source_object_vocab = self._build_vocab(self.planner_labels, "source_object")
        self.target_object_vocab = self._build_vocab(self.planner_labels, "target_object")
        self.target_region_vocab = self._build_vocab(self.planner_labels, "target_region")

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

    def _load_planner_labels(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda row: row["flat_index"])
        expected = self.replay_buffer["action"].shape[0]
        if len(rows) != expected:
            raise ValueError(f"Planner label count {len(rows)} does not match replay buffer length {expected}.")

        expected_episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        expected_flat_indices = np.arange(expected, dtype=np.int64)
        actual_flat_indices = np.asarray([int(row["flat_index"]) for row in rows], dtype=np.int64)
        if not np.array_equal(actual_flat_indices, expected_flat_indices):
            raise ValueError("Planner labels must contain contiguous flat_index values matching replay buffer transitions.")

        expected_episode_ids = np.zeros(expected, dtype=np.int64)
        expected_timesteps = np.zeros(expected, dtype=np.int64)
        start = 0
        for episode_id, end in enumerate(expected_episode_ends):
            end = int(end)
            expected_episode_ids[start:end] = episode_id
            expected_timesteps[start:end] = np.arange(end - start, dtype=np.int64)
            start = end

        actual_episode_ids = np.asarray([int(row["episode_id"]) for row in rows], dtype=np.int64)
        actual_timesteps = np.asarray([int(row["timestep"]) for row in rows], dtype=np.int64)
        if not np.array_equal(actual_episode_ids, expected_episode_ids):
            raise ValueError("Planner label episode_id values do not align with replay buffer episode boundaries.")
        if not np.array_equal(actual_timesteps, expected_timesteps):
            raise ValueError("Planner label timestep values do not align with replay buffer episode-relative timesteps.")
        return rows

    @staticmethod
    def _build_vocab(rows, key: str):
        vocab = {None: 0}
        for row in rows:
            value = row.get(key)
            if value not in vocab:
                vocab[value] = len(vocab)
        return vocab

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

    def _encode_label(self, row: Dict):
        return {
            "stage_id": np.array(self.stage_vocab[row.get("stage")], dtype=np.int64),
            "stage_type_id": np.array(self.stage_type_vocab[row.get("stage_type")], dtype=np.int64),
            "source_object_id": np.array(self.source_object_vocab[row.get("source_object")], dtype=np.int64),
            "target_object_id": np.array(self.target_object_vocab[row.get("target_object")], dtype=np.int64),
            "target_region_id": np.array(self.target_region_vocab[row.get("target_region")], dtype=np.int64),
            "is_terminal_stage": np.array(int(bool(row.get("is_terminal_stage"))), dtype=np.int64),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        # Use the last valid replay-buffer transition in the sampled chunk as the planner supervision target.
        end_flat_index = int(buffer_end_idx - 1)
        planner_row = self.planner_labels[end_flat_index]
        planner = self._encode_label(planner_row)

        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
            "planner": planner,
        }
        return dict_apply(data, torch.from_numpy)
