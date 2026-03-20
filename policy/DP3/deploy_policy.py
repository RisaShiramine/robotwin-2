# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib
import dill

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *
from diffusion_policy_3d.dataset.robot_dataset import inspect_planner_tokens


def encode_obs(observation):  # Post-Process Observation
def encode_obs(observation, planner_tokens=None):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']

    dynamic_tokens = planner_tokens or {}
    for key in ('stage_id', 'source_id', 'target_id'):
        if key in observation:
            dynamic_tokens[key] = observation[key]
    if dynamic_tokens:
        for key in ('stage_id', 'source_id', 'target_id'):
            if key in dynamic_tokens:
                obs[key] = np.array(dynamic_tokens[key], dtype=np.int64)
    return obs






def resolve_eval_zarr_path(cfg, usr_args):
    setting = usr_args.get("ckpt_setting") or usr_args.get("task_config") or cfg.get("setting", None)
    expert_data_num = usr_args.get("expert_data_num", cfg.get("expert_data_num", None))
    task_name = usr_args.get("task_name") or cfg.get("task_name", None)
    if task_name is None or setting is None or expert_data_num is None:
        return None
    return f"../../../data/{task_name}-{setting}-{expert_data_num}.zarr"


def maybe_inspect_eval_planner_tokens(cfg, usr_args):
    zarr_path = resolve_eval_zarr_path(cfg, usr_args)
    if not zarr_path:
        return False, {}
    try:
        return inspect_planner_tokens(zarr_path)
    except Exception:
        return False, {}

def resolve_ckpt_path(cfg, usr_args):
    if not cfg.policy.use_pc_color:
        return pathlib.Path(os.path.join(
            parent_directory,
            '3D-Diffusion-Policy',
            f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt",
        ))
    return pathlib.Path(os.path.join(
        parent_directory,
        '3D-Diffusion-Policy',
        f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_w_rgb_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt",
    ))


def align_policy_cfg_with_checkpoint(cfg, usr_args):
    ckpt_file = resolve_ckpt_path(cfg, usr_args)
    if not ckpt_file.is_file():
        return

    payload = torch.load(ckpt_file.open('rb'), pickle_module=dill, map_location='cpu')
    ckpt_cfg = payload.get('cfg')
    if ckpt_cfg is None:
        return

    OmegaConf.set_struct(cfg, False)
    cfg.policy.use_planner_tokens = bool(ckpt_cfg.policy.get('use_planner_tokens', False))
    cfg.policy.planner_embed_dim = int(ckpt_cfg.policy.get('planner_embed_dim', cfg.policy.planner_embed_dim))
    cfg.policy.planner_vocab_sizes = OmegaConf.to_container(ckpt_cfg.policy.get('planner_vocab_sizes', cfg.policy.planner_vocab_sizes), resolve=True)
    if 'shape_meta' in ckpt_cfg:
        cfg.shape_meta = ckpt_cfg.shape_meta
        cfg.policy.shape_meta = ckpt_cfg.shape_meta
    OmegaConf.set_struct(cfg, True)

def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.setting = usr_args.get("ckpt_setting", cfg.get("setting", None))
    cfg.policy.use_pc_color = usr_args['use_rgb']
    has_planner_tokens, vocab_sizes = maybe_inspect_eval_planner_tokens(cfg, usr_args)
    cfg.policy.use_planner_tokens = bool(has_planner_tokens)
    if has_planner_tokens and vocab_sizes:
        cfg.policy.planner_vocab_sizes = vocab_sizes
    align_policy_cfg_with_checkpoint(cfg, usr_args)
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    return DP3_Model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    planner_tokens = None
    if hasattr(model, "env_runner") and getattr(model.env_runner, "planner_controller", None) is not None:
        planner_tokens = model.env_runner.planner_controller.current_tokens()
    obs = encode_obs(observation, planner_tokens=planner_tokens)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        planner_tokens = None
        if hasattr(model, "env_runner") and getattr(model.env_runner, "planner_controller", None) is not None:
            next_obs = encode_obs(observation)
            model.env_runner.planner_controller.maybe_advance(next_obs)
            planner_tokens = model.env_runner.planner_controller.current_tokens()
        obs = encode_obs(observation, planner_tokens=planner_tokens)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
    if getattr(model.env_runner, "planner_controller", None) is not None:
        model.env_runner.planner_controller.reset()