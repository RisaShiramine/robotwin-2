# import packages and module here
import sys, os, json
import torch
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# Load precomputed embedding lookup table once at import time
_EMBEDDING_MAPPING_PATH = "/mnt/hdd/Project/RoboticsDiffusionTransformer/data/embeddings/robotwin/mapping.json"
with open(_EMBEDDING_MAPPING_PATH, "r") as _f:
    _EMBEDDING_MAPPING: dict = json.load(_f)


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def get_model(usr_args):  # keep
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    rdt = RDT(
        os.path.join(
            "/mnt/hdd/Project/RoboticsDiffusionTransformer/checkpoints",
            f"{model_name}/checkpoint-{checkpoint_id}/model.safetensors",
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    )
    return rdt


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    # Use per-episode instruction from TASK_ENV (contains color/order info)
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        print(f"\033[96m[Instruction] {instruction}\033[0m")
        # Look up precomputed embedding; fall back to T5 encoding if not found
        if instruction in _EMBEDDING_MAPPING:
            embed_path = _EMBEDDING_MAPPING[instruction]
            emb = torch.load(embed_path, map_location="cpu", weights_only=False)
            if emb.dim() == 2:          # [seq_len, 4096] -> [1, seq_len, 4096]
                emb = emb.unsqueeze(0)
            model.lang_embeddings = emb
            print(f"\033[92m[Embedding] Loaded from cache: {os.path.basename(embed_path)}\033[0m")
        else:
            print(f"\033[93m[Embedding] Not found in mapping, falling back to T5 encoding\033[0m")
            model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    # ── SubGoalTokenizer: compute (arm, color, step) for current chunk ──
    # step_seg mirrors training: min(take_action_cnt * 3 // step_lim, 2)
    # arm is unknown at eval time (zero embedding → no-op in cross-attn)
    # color: step 0=red(0), 1=green(1), 2=blue(2)
    _step_lim = getattr(TASK_ENV, 'step_lim', 1200)
    _cnt      = getattr(TASK_ENV, 'take_action_cnt', 0)
    _step_seg = min(_cnt * 3 // max(_step_lim, 1), 2)
    subgoal = {
        "arm":   torch.tensor([2], dtype=torch.long),   # 2 = unknown
        "color": torch.tensor([_step_seg], dtype=torch.long),   # 0=red 1=green 2=blue
        "step":  torch.tensor([_step_seg], dtype=torch.long),
    }

    actions = model.get_action(subgoal=subgoal)[:model.rdt_step, :]  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
