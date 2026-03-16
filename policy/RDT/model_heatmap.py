#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import json
import sys

parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import os

import argparse

import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

import sys, os

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent, "models"))

from scripts.agilex_model import create_model
from multimodal_encoder.t5_encoder import T5Embedder

global_path = parent_dir.parent

# ===== ATTENTION VISUALIZATION: global state =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ATTENTION_MAPS = []
_ATTN_VIS_STEP = [0]  # mutable counter
_ATTN_PATCHED = [False]  # applied only once


def _make_patched_cross_attn_forward():
    def patched_forward(self, x, c, mask=None):
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L).expand(-1, -1, N, -1)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
        attn = attn.softmax(dim=-1)
        _ATTENTION_MAPS.append(attn.detach().cpu())
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x
    return patched_forward


def _apply_attn_patch(policy):
    # Patch the actual CrossAttention class used by model instances.
    # We traverse model submodules to find the real class object,
    # avoiding sys.modules key mismatch (rdt.blocks vs models.rdt.blocks).
    if _ATTN_PATCHED[0]:
        return
    new_fwd = _make_patched_cross_attn_forward()
    patched_classes = set()
    # policy may be RDTRunner wrapping an inner .policy
    root = getattr(policy, 'policy', policy)
    for m in root.modules():
        cls = type(m)
        if cls.__name__ == 'CrossAttention' and cls not in patched_classes:
            cls.forward = new_fwd
            patched_classes.add(cls)
    if patched_classes:
        _ATTN_PATCHED[0] = True
        print(f'[Attn Vis] CrossAttention patched ({len(patched_classes)} class variant(s))')
    else:
        print('[Attn Vis] WARNING: No CrossAttention found in model!')
# ===== END GLOBAL STATE =====


class RDT:

    def __init__(
        self,
        pretrained_model_name_or_path,
        task_name,
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    ):
        # set path
        current_file = Path(__file__)
        self.global_path = current_file.parent.parent
        # load the config
        self.config = {
            "episode_len": 10000,  # args.max_publish_step
            "state_dim": left_arm_dim + 1 + right_arm_dim +
            1,  # 14 dims action:[left joint angles,left gripper,right joint angles,right gripper]
            "chunk_size": 64,  # args.chunk_size
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        }
        # setup config
        self.args = {
            "max_publish_step": 10000,  # Maximum number of action publishing steps
            "seed": None,  # Random seed
            "ctrl_freq": 25,  # The control frequency of the robot
            "chunk_size": 64,  # Action chunk size
            # 'disable_puppet_arm': False,  # Whether to disable the puppet arm
            "config_path": os.path.join(self.global_path, "RDT/configs/base.yaml"),
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
        }

        # Load rdt model
        self.left_arm_dim, self.right_arm_dim = left_arm_dim, right_arm_dim
        self.policy = self.make_policy(self.args)
        self.max_publish_step = self.config["episode_len"]
        self.chunk_size = self.config["chunk_size"]
        self.task_name = task_name
        self.observation_window = None
        self.img_size = (640, 480)
        self.text_encoder = None  # Lazy load only when needed
        self.tokenizer = None
        self.rdt_step = rdt_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language_embed(self):
        """Lazy load T5 model only when needed (not using precomputed embeddings)"""
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            return  # Already loaded
        
        GPU = 0
        MODEL_PATH = os.path.join(self.global_path, "weights/RDT/t5-v1_1-xxl")
        CONFIG_PATH = os.path.join(self.global_path, "RDT/configs/base.yaml")
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        device = torch.device(f"cuda:{GPU}")
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=device,
            use_offload_folder=None,
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        self.text_encoder.eval()

    # set language randomly
    def random_set_language(self, instruction=None):
        assert instruction is not None, "Missing input instruction"
        self.set_language_instruction(instruction)

    # encoding language
    def set_language_instruction(self, language_instruction, save_dir=None, task_name=None):
        assert ((save_dir is None) ^ (task_name is None)) == False, "input error"

        if os.path.isfile(language_instruction):
            lang_dict = torch.load(language_instruction)
            # Handle both dict format and raw tensor
            if isinstance(lang_dict, dict):
                print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
                self.lang_embeddings = lang_dict["embeddings"]
            else:
                # Raw tensor (precomputed embedding)
                print(f"Loading precomputed embedding from {language_instruction}")
                self.lang_embeddings = lang_dict
            print("loading instruction from pre-embed path")
        else:
            # Load T5 model if not already loaded (for real-time encoding)
            if self.text_encoder is None:
                self.set_language_embed()
            device = next(self.text_encoder.parameters()).device
            with torch.no_grad():
                tokens = self.tokenizer(
                    language_instruction,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )["input_ids"].to(device)
                tokens = tokens.view(1, -1)
                output = self.text_encoder(tokens)
                pred = output.last_hidden_state.detach().cpu()

            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{task_name}.pt")
                torch.save({
                    "name": task_name,
                    "instruction": language_instruction,
                    "embeddings": pred,
                }, save_path)

            del tokens, output
            torch.cuda.empty_cache()
            self.lang_embeddings = pred

        print(f"successfully set instruction: {language_instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        # JPEG transformation
        # Align with training
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img

        def resize_img(img, size):
            return cv2.resize(img, size)

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            # Append the first dummy image
            self.observation_window.append({
                "qpos": None,
                "images": {
                    self.config["camera_names"][0]: None,
                    self.config["camera_names"][1]: None,
                    self.config["camera_names"][2]: None,
                },
            })

        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        # img resize
        img_front = resize_img(img_front, self.img_size)
        img_left = resize_img(img_left, self.img_size)
        img_right = resize_img(img_right, self.img_size)
        # img jprg encoding
        img_front = jpeg_mapping(img_front)
        img_left = jpeg_mapping(img_left)
        img_right = jpeg_mapping(img_right)

        qpos = np.array(puppet_arm)
        qpos = torch.from_numpy(qpos).float().cuda()
        self.observation_window.append({
            "qpos": qpos,
            "images": {
                self.config["camera_names"][0]: img_front,
                self.config["camera_names"][1]: img_right,
                self.config["camera_names"][2]: img_left,
            },
        })

    def get_action(self, img_arr=None, state=None):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        with torch.inference_mode():
            action_buffer = inference_fn(self.config, self.policy, self.lang_embeddings, self.observation_window).copy()

        return action_buffer

    def reset_obsrvationwindows(self):
        self.lang_embeddings = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Initialize the model
    def make_policy(self, args):
        with open(args["config_path"], "r") as fp:
            config_base_yaml = yaml.safe_load(fp)

        # Auto-detect architecture from checkpoint's config.json (handles 170M vs 1B,
        # SubGoalTokenizer, QFormer, LangFiLM, ArmOrderPrefix, etc.)
        import json
        ckpt_dir = os.path.dirname(args["pretrained_model_name_or_path"])
        ckpt_config_path = os.path.join(ckpt_dir, "config.json")
        if os.path.isfile(ckpt_config_path):
            with open(ckpt_config_path, "r") as fp:
                ckpt_config = json.load(fp)
            _MODEL_KEYS = [
                "rdt", "subgoal_tokenizer", "qformer", "lang_film",
                "arm_order_prefix", "aux_arm_order",
            ]
            for key in _MODEL_KEYS:
                if key in ckpt_config:
                    if key not in config_base_yaml["model"]:
                        config_base_yaml["model"][key] = {}
                    config_base_yaml["model"][key].update(ckpt_config[key])
            print(f"[make_policy] Overriding model arch from checkpoint config.json "
                  f"(rdt={ckpt_config.get('rdt')}, "
                  f"subgoal={ckpt_config.get('subgoal_tokenizer')}, "
                  f"qformer={ckpt_config.get('qformer')})")

        args["config"] = config_base_yaml
        args["config"]["arm_dim"] = {
            "left_arm_dim": self.left_arm_dim,
            "right_arm_dim": self.right_arm_dim,
        }
        # pretrained_text_encoder_name_or_path = "weights/RDT/t5-v1_1-xxl"
        pretrained_vision_encoder_name_or_path = os.path.join(self.global_path, "weights/RDT/siglip-so400m-patch14-384")
        model = create_model(
            args=args["config"],
            dtype=torch.bfloat16,
            pretrained=args["pretrained_model_name_or_path"],
            # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            control_frequency=args["ctrl_freq"],
        )

        _apply_attn_patch(model)  # patch CrossAttention after model/blocks are loaded
        return model


# RDT inference
def inference_fn(config, policy, lang_embeddings, observation_window):

    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]["images"][config["camera_names"][0]],
            observation_window[-2]["images"][config["camera_names"][1]],
            observation_window[-2]["images"][config["camera_names"][2]],
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]["qpos"]
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = (policy.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy())
        # print(f"inference_actions: {actions.squeeze()}")

        # print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        # ===== SAVE ATTENTION HEATMAP =====
        if len(_ATTENTION_MAPS) > 0:
            try:
                final_attn = _ATTENTION_MAPS[-1]  # (B, H, Q_len, K_len)
                lang_len = lang_embeddings.shape[1]
                img_attn = final_attn[..., lang_len:]  # (B, H, Q_len, Img_len)
                num_patches = 729  # 27x27 for SigLIP-384
                # images order: [front_t-1, right_t-1, left_t-1, front_t, right_t, left_t]
                # front camera at current timestep is index 3
                front_attn = img_attn[..., 3 * num_patches: 4 * num_patches]
                front_attn_mean = front_attn.mean(dim=(0, 1))  # (Q_len, 729)
                raw_front = images[3] if images[3] is not None else images[0]
                if raw_front is not None:
                    vis_img = raw_front.resize((384, 384))
                    vis_np = np.array(vis_img)
                    grid_size = 27
                    _out_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'eval_attention_maps'
                    )
                    os.makedirs(_out_dir, exist_ok=True)
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    for ki, step_idx in enumerate([0, 20, 40, 63]):
                        q_idx = 3 + step_idx  # offset: t(1)+freq(1)+state(1)=3
                        ax = axes[ki]
                        if q_idx < front_attn_mean.shape[0]:
                            amap = front_attn_mean[q_idx].view(grid_size, grid_size).float().numpy()
                            if amap.max() > amap.min():
                                amap = (amap - amap.min()) / (amap.max() - amap.min())
                            amap_up = cv2.resize(amap, (384, 384), interpolation=cv2.INTER_NEAREST)
                            heatmap = cv2.applyColorMap(np.uint8(255 * amap_up), cv2.COLORMAP_JET)
                            overlay = cv2.addWeighted(vis_np, 0.5, heatmap, 0.5, 0)
                            ax.imshow(overlay)
                        else:
                            ax.imshow(vis_np)
                        ax.set_title(f'Chunk action step {step_idx}')
                        ax.axis('off')
                    _save_path = os.path.join(_out_dir, f'chunk_{_ATTN_VIS_STEP[0]:04d}.png')
                    plt.savefig(_save_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'[Attn Vis] Saved -> {_save_path}')
                    _ATTN_VIS_STEP[0] += 1
            except Exception as _vis_err:
                print(f'[Attn Vis] Error: {_vis_err}')
            _ATTENTION_MAPS.clear()
        # ===== END SAVE =====
        return actions
