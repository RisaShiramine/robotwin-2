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
import json

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(parent_directory, 'scripts'))

from dp3_policy import *
from diffusion_policy_3d.dataset.robot_dataset import inspect_planner_tokens
from diffusion_policy_3d.env_runner.planner_controller import PlannerTokenStateMachine
from planner_decomposition_utils import normalize_decomposition_row






def planner_debug_once(model, key, message, force=False):
    runtime_cfg = getattr(model, "planner_runtime", None)
    if runtime_cfg is None:
        planner_debug_log(model, message, force=force)
        return
    logged = runtime_cfg.setdefault("logged_messages", set())
    if key in logged:
        return
    logged.add(key)
    planner_debug_log(model, message, force=force)

def planner_debug_enabled(model):
    runtime_cfg = getattr(model, "planner_runtime", None) or {}
    return bool(runtime_cfg.get("planner_debug", True))


def planner_debug_log(model, message, force=False):
    if force or planner_debug_enabled(model):
        print(f"[Planner][Runtime] {message}")


def get_policy_planner_enabled(model):
    policy = getattr(model, "policy", None)
    return bool(getattr(policy, "use_planner_tokens", False))


def log_controller_state(model, prefix):
    controller = getattr(model.env_runner, "planner_controller", None)
    if controller is None:
        return
    tokens = controller.current_tokens()
    planner_debug_log(model, f"{prefix}: {controller.describe_stage()} tokens={tokens}")

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










def normalize_object_key(text):
    if text is None:
        return None
    return normalize_instruction_key(text).replace("_", " ")


def is_stack_blocks_task(task_env):
    task_name = task_env.__class__.__name__.lower()
    return "stack" in task_name and "block" in task_name


def get_stack_center_xy(task_env):
    if hasattr(task_env, "block1_target_pose") and len(task_env.block1_target_pose) >= 2:
        return np.array(task_env.block1_target_pose[:2], dtype=np.float32)
    return np.array([0.0, -0.13], dtype=np.float32)


def all_grippers_open(task_env):
    left_open = task_env.is_left_gripper_open() if hasattr(task_env, "is_left_gripper_open") else True
    right_open = task_env.is_right_gripper_open() if hasattr(task_env, "is_right_gripper_open") else True
    return bool(left_open and right_open)


def resolve_block_actor(task_env, object_name):
    key = normalize_object_key(object_name)
    if key is None:
        return None
    key = key.replace(" block", "")
    direct_attr = key.replace(" ", "_") + "_block"
    if hasattr(task_env, direct_attr):
        return getattr(task_env, direct_attr)

    if hasattr(task_env, "blocks") and hasattr(task_env, "block_color_names"):
        color_names = [normalize_object_key(name) for name in task_env.block_color_names]
        lookup_key = f"{key} block" if f"{key} block" in color_names else key
        if lookup_key in color_names:
            return task_env.blocks[color_names.index(lookup_key)]

    if hasattr(task_env, "block_color_name") and hasattr(task_env, "block1"):
        single_name = normalize_object_key(task_env.block_color_name)
        if key == single_name or key == f"{single_name} block":
            return task_env.block1

    return None


def actor_pose_xyz(actor):
    if actor is None:
        return None
    return np.asarray(actor.get_pose().p, dtype=np.float32)


def stage_target_pose(task_env, stage):
    target_support = normalize_object_key(stage.get("target_support"))
    target_region = normalize_object_key(stage.get("target_region"))
    target_object = normalize_object_key(stage.get("target_object"))

    if target_support in {"table", "center", None} or target_region == "center":
        center_xy = get_stack_center_xy(task_env)
        return np.array([center_xy[0], center_xy[1], None], dtype=object)

    support_actor = resolve_block_actor(task_env, target_support or target_object)
    support_pose = actor_pose_xyz(support_actor)
    if support_pose is None:
        return None
    return np.array([support_pose[0], support_pose[1], support_pose[2] + 0.05], dtype=object)


def is_move_stage_complete(task_env, stage):
    source_actor = resolve_block_actor(task_env, stage.get("source_object") or stage.get("target_object"))
    source_pose = actor_pose_xyz(source_actor)
    target_pose = stage_target_pose(task_env, stage)
    if source_pose is None or target_pose is None:
        return False

    xy_ok = np.all(np.abs(source_pose[:2] - np.asarray(target_pose[:2], dtype=np.float32)) < np.array([0.03, 0.03]))
    return bool(xy_ok and all_grippers_open(task_env))


def is_stack_stage_complete(task_env, stage):
    source_actor = resolve_block_actor(task_env, stage.get("source_object"))
    support_actor = resolve_block_actor(task_env, stage.get("target_support") or stage.get("target_object"))
    source_pose = actor_pose_xyz(source_actor)
    support_pose = actor_pose_xyz(support_actor)
    if source_pose is None or support_pose is None:
        return False

    expected_pose = np.array([support_pose[0], support_pose[1], support_pose[2] + 0.05], dtype=np.float32)
    eps = np.array([0.025, 0.025, 0.015], dtype=np.float32)
    return bool(np.all(np.abs(source_pose - expected_pose) < eps) and all_grippers_open(task_env))


def build_stack_blocks_completion_rule(task_env):
    def completion_rule(obs, stage):
        runtime_stage = dict(stage)
        for runtime_key, base_key in (
            ("runtime_stage", "stage"),
            ("runtime_stage_type", "stage_type"),
            ("runtime_target_object", "target_object"),
            ("runtime_support_object", "support_object"),
            ("runtime_target_support", "target_support"),
            ("runtime_target_region", "target_region"),
            ("runtime_spatial_relation", "spatial_relation"),
        ):
            if runtime_key in stage:
                runtime_stage[base_key] = stage.get(runtime_key)

        stage_type = normalize_object_key(runtime_stage.get("stage_type")) or "unknown"
        if stage_type in {"move to region", "move_to_region", "establish base", "establish_base"}:
            return is_move_stage_complete(task_env, runtime_stage)
        if stage_type in {"stack on support", "stack_on_support"}:
            return is_stack_stage_complete(task_env, runtime_stage)
        return False

    return completion_rule

def normalize_instruction_key(text):
    if text is None:
        return None
    return " ".join(str(text).strip().lower().split())


def collapse_execution_stages_for_runtime(stages):
    execution_stage_types = {
        "align_for_pick",
        "grasp_object",
        "transport_in_air",
        "place_release",
        "retreat_and_reset",
    }
    if not stages or not any(str(stage.get("stage_type")) in execution_stage_types for stage in stages):
        return stages

    runtime_stages = []
    for stage in stages:
        stage_type = str(stage.get("stage_type") or "")
        if stage_type != "place_release":
            continue

        source_object = stage.get("source_object")
        support_object = stage.get("support_object")
        target_region = stage.get("target_region") or "center"
        target_support = stage.get("target_support")
        is_table_stage = support_object in {None, "table", "center", "table_center"} or target_support in {None, "table", "center", "table_center"}

        if is_table_stage:
            runtime_stages.append(
                {
                    **stage,
                    "runtime_stage": f"move_{source_object}_to_{target_region}",
                    "runtime_stage_type": "move_to_region",
                    "runtime_action_type": "move",
                    "runtime_target_object": None,
                    "runtime_support_object": None,
                    "runtime_target_support": None,
                    "runtime_target_region": target_region,
                    "runtime_spatial_relation": None,
                }
            )
        else:
            runtime_stages.append(
                {
                    **stage,
                    "runtime_stage": f"stack_{source_object}_on_{support_object}",
                    "runtime_stage_type": "stack_on_support",
                    "runtime_action_type": "stack",
                    "runtime_target_object": support_object,
                    "runtime_support_object": support_object,
                    "runtime_target_support": support_object,
                    "runtime_target_region": target_region,
                    "runtime_spatial_relation": "over",
                }
            )

    return runtime_stages or stages


def load_planner_stages_from_decomposition(path, instruction):
    if not path or not instruction:
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    target = normalize_instruction_key(instruction)
    records = []
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                records.extend(value)
    elif isinstance(payload, list):
        records = payload

    for record in records:
        record_instruction = record.get("instruction")
        decomposition = record.get("decomposition", {})
        decomposition_instruction = decomposition.get("instruction")
        if target in {normalize_instruction_key(record_instruction), normalize_instruction_key(decomposition_instruction)}:
            normalized_record = normalize_decomposition_row(
                {
                    "split": record.get("split"),
                    "index": record.get("index"),
                    "instruction": record_instruction or decomposition_instruction,
                    "decomposition": decomposition,
                },
                expand_stack_execution=True,
            )
            stages = normalized_record.get("decomposition", {}).get("stages", [])
            if not stages:
                return None
            return collapse_execution_stages_for_runtime(stages)
    return None


def maybe_init_planner_controller(TASK_ENV, model):
    if getattr(model.env_runner, "planner_controller", None) is not None:
        return

    runtime_cfg = getattr(model, "planner_runtime", None)
    if not runtime_cfg:
        planner_debug_once(model, "missing_runtime", "planner_runtime is missing; planner controller will not be created")
        return

    vocab_path = runtime_cfg.get("planner_vocab_path")
    decomposition_path = runtime_cfg.get("planner_decomposition_path")
    if not vocab_path or not decomposition_path:
        planner_debug_once(
            model,
            "missing_planner_paths",
            f"planner controller disabled because planner_vocab_path={vocab_path} planner_decomposition_path={decomposition_path}",
            force=True,
        )
        return

    instruction = TASK_ENV.get_instruction()
    if not instruction:
        planner_debug_once(model, "missing_instruction", "TASK_ENV.get_instruction() is empty; cannot initialize planner controller", force=True)
        return

    stages = load_planner_stages_from_decomposition(decomposition_path, instruction)
    if not stages:
        planner_debug_once(model, f"unmatched_instruction::{instruction}", f"no planner stages matched instruction: {instruction}", force=True)
        return

    completion_rule = build_stack_blocks_completion_rule(TASK_ENV) if is_stack_blocks_task(TASK_ENV) else None
    controller = PlannerTokenStateMachine(
        vocab_path=vocab_path,
        stages=stages,
        completion_rule=completion_rule,
        debug=bool(runtime_cfg.get("planner_debug", True)),
        debug_prefix=f"[Planner][{TASK_ENV.__class__.__name__}]",
    )
    model.set_planner_controller(controller)
    planner_debug_log(model, f"controller initialized for instruction: {instruction}", force=True)
    log_controller_state(model, "initial stage")

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
    planner_debug_log(DP3_Model, f"checkpoint planner branch enabled={get_policy_planner_enabled(DP3_Model)}", force=True)
    DP3_Model.planner_runtime = {
        "planner_vocab_path": usr_args.get("planner_vocab_path"),
        "planner_decomposition_path": usr_args.get("planner_decomposition_path"),
        "planner_debug": usr_args.get("planner_debug", True),
    }
    return DP3_Model


def eval(TASK_ENV, model, observation):
    maybe_init_planner_controller(TASK_ENV, model)
    planner_tokens = None
    if get_policy_planner_enabled(model) and getattr(model.env_runner, "planner_controller", None) is None:
        planner_debug_once(model, "missing_runtime_controller", "planner-conditioned checkpoint loaded, but no runtime planner controller is active", force=True)
    if hasattr(model, "env_runner") and getattr(model.env_runner, "planner_controller", None) is not None:
        planner_tokens = model.env_runner.planner_controller.current_tokens()
        log_controller_state(model, "before action chunk")
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
        planner_tokens = None
        if hasattr(model, "env_runner") and getattr(model.env_runner, "planner_controller", None) is not None:
            next_obs = encode_obs(observation)
            advanced = model.env_runner.planner_controller.maybe_advance(next_obs)
            planner_tokens = model.env_runner.planner_controller.current_tokens()
            if advanced:
                log_controller_state(model, "after advance")
        obs = encode_obs(observation, planner_tokens=planner_tokens)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
    if getattr(model.env_runner, "planner_controller", None) is not None:
        model.env_runner.planner_controller.reset()
    elif get_policy_planner_enabled(model):
        planner_debug_once(model, "reset_without_controller", "reset with planner-conditioned checkpoint but no controller attached", force=True)
