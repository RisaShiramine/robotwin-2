import os
import time

import wandb
import numpy as np
import torch
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import pdb
from queue import deque


def _pc_rgb_to_01(rgb: np.ndarray) -> np.ndarray:
    """
    Normalize rgb to [0, 1].
    Supports either [0,1] or [-1,1].
    """
    rgb = rgb.astype(np.float32)
    if rgb.min() < -0.2:
        rgb = (rgb + 1.0) / 2.0
    return np.clip(rgb, 0.0, 1.0)


def summarize_pointcloud(point_cloud: np.ndarray, prefix: str = ""):
    xyz = point_cloud[:, :3]
    rgb = _pc_rgb_to_01(point_cloud[:, 3:6])
    print(
        f"{prefix} shape={point_cloud.shape} "
        f"xyz_min={xyz.min(axis=0)} xyz_max={xyz.max(axis=0)} "
        f"rgb_min={rgb.min(axis=0)} rgb_max={rgb.max(axis=0)} "
        f"rgb_mean={rgb.mean(axis=0)}"
    )


def classify_color_points(
    point_cloud: np.ndarray,
    sat_thresh: float = 0.10,
    red_margin: float = 0.05,
    green_margin: float = 0.05,
    blue_margin: float = 0.05,
):
    """
    point_cloud: [N, 6] -> [x,y,z,r,g,b]
    returns:
        is_red, is_green, is_blue, is_neutral
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[1] >= 6, \
        f"Expected [N,6+], got {point_cloud.shape}"

    rgb = _pc_rgb_to_01(point_cloud[:, 3:6])
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    rgb_max = np.max(rgb, axis=1)
    rgb_min = np.min(rgb, axis=1)
    sat = rgb_max - rgb_min

    is_colored = sat > sat_thresh

    is_red = is_colored & (r > g + red_margin) & (r > b + red_margin)
    is_green = is_colored & (g > r + green_margin) & (g > b + green_margin)
    is_blue = is_colored & (b > r + blue_margin) & (b > g + blue_margin)

    is_neutral = ~(is_red | is_green | is_blue)
    return is_red, is_green, is_blue, is_neutral


def debug_color_stats(point_cloud: np.ndarray, prefix: str = ""):
    is_red, is_green, is_blue, is_neutral = classify_color_points(point_cloud)
    print(
        f"{prefix} total={len(point_cloud)} "
        f"red={is_red.sum()} green={is_green.sum()} "
        f"blue={is_blue.sum()} neutral={is_neutral.sum()}"
    )


def check_red_on_bottom(point_cloud: np.ndarray, z_thresh: float = 0.015) -> bool:
    """
    检查红方块是否被平稳放在最底部（和底部的空隙极小）
    """
    is_red, _, _, _ = classify_color_points(point_cloud)
    if is_red.sum() < 5:
        return False
    
    # 采用 5% 和 1% 分位数防止深度相机的少量噪点/飞点造成误判
    red_z_min = np.percentile(point_cloud[is_red, 2], 5)
    global_z_min = np.percentile(point_cloud[:, 2], 1)
    
    return (red_z_min - global_z_min) < z_thresh


def check_green_on_red(point_cloud: np.ndarray, dist_thresh: float = 0.025) -> bool:
    """
    检查绿方块是否成功堆叠在红方块上面
    """
    is_red, is_green, _, _ = classify_color_points(point_cloud)
    if is_red.sum() < 5 or is_green.sum() < 5:
        return False
        
    red_pts = point_cloud[is_red, :3]
    green_pts = point_cloud[is_green, :3]
    
    # 绿方块的重心应高于红方块
    if green_pts[:, 2].mean() <= red_pts[:, 2].mean():
        return False
        
    # 【新增对齐检测】绿方块和红方块在水平面上不能偏离太远，防止侧边滑落误判
    red_xy = red_pts[:, :2].mean(axis=0)
    green_xy = green_pts[:, :2].mean(axis=0)
    if np.linalg.norm(red_xy - green_xy) > 0.04:
        return False
        
    # 随机下采样加速计算
    if len(red_pts) > 300:
        red_pts = red_pts[np.random.choice(len(red_pts), 300, replace=False)]
    if len(green_pts) > 300:
        green_pts = green_pts[np.random.choice(len(green_pts), 300, replace=False)]
        
    dists = np.sqrt(((red_pts[:, np.newaxis, :] - green_pts[np.newaxis, :, :])**2).sum(axis=2))
    min_dist = dists.min()

    return min_dist < dist_thresh


def check_robot_away_from_block(point_cloud: np.ndarray, block_mask: np.ndarray, safe_dist: float = 0.035, min_robot_pts_threshold: int = 15) -> bool:
    """
    【极速检测】基于 Bounding Box 计算。只要夹爪张开并脱离方块表面 3.5cm 即视为释放！
    """
    if block_mask.sum() < 5:
        return True
        
    block_pts = point_cloud[block_mask, :3]
    
    _, _, _, is_neutral = classify_color_points(point_cloud)
    global_z_min = np.percentile(point_cloud[:, 2], 1)
    
    # 避开桌面点，要求高度大于桌面 2cm 的纯黑色点才可能是夹爪
    possible_robot_mask = is_neutral & (point_cloud[:, 2] > global_z_min + 0.02)
    
    if possible_robot_mask.sum() < min_robot_pts_threshold:
        return True 
        
    robot_pts = point_cloud[possible_robot_mask, :3]
    
    # 构建方块的 3D 紧凑包围盒，并向外扩展 safe_dist (3.5cm)
    block_min = np.percentile(block_pts, 1, axis=0)
    block_max = np.percentile(block_pts, 99, axis=0)
    
    box_min = block_min - safe_dist
    box_max = block_max + safe_dist
    
    # 检测落在这个紧贴盒子里面的机械臂点数
    in_box_mask = (
        (robot_pts[:, 0] >= box_min[0]) & (robot_pts[:, 0] <= box_max[0]) &
        (robot_pts[:, 1] >= box_min[1]) & (robot_pts[:, 1] <= box_max[1]) &
        (robot_pts[:, 2] >= box_min[2]) & (robot_pts[:, 2] <= box_max[2])
    )
    
    return in_box_mask.sum() < min_robot_pts_threshold


def mask_green_blue_points_for_policy(
    point_cloud: np.ndarray,
    min_keep_points: int = 64,
    remove_mode: str = 'gb',
):
    assert point_cloud.ndim == 2 and point_cloud.shape[1] >= 6, \
        f"Expected [N,6+], got {point_cloud.shape}"

    out = point_cloud.copy()

    if remove_mode == 'none':
        return out

    is_red, is_green, is_blue, is_neutral = classify_color_points(out)

    if remove_mode == 'gb':
        remove_mask = is_green | is_blue
        keep_mask = is_red | is_neutral
    elif remove_mode == 'rb':
        remove_mask = is_red | is_blue
        keep_mask = is_green | is_neutral
    elif remove_mode == 'b':
        remove_mask = is_blue
        keep_mask = is_red | is_green | is_neutral
    elif remove_mode == 'r':
        remove_mask = is_red
        keep_mask = is_green | is_blue | is_neutral
    else:
        return out

    keep_idx = np.flatnonzero(keep_mask)
    remove_idx = np.flatnonzero(remove_mask)

    if len(keep_idx) < min_keep_points:
        return out

    if len(remove_idx) == 0:
        return out

    fill_idx = keep_idx[np.arange(len(remove_idx)) % len(keep_idx)]
    out[remove_idx] = out[fill_idx]

    return out


def mask_green_blue_sequence_for_policy(
    point_cloud_seq: np.ndarray,
    min_keep_points: int = 64,
    remove_mode: str = 'gb',
):
    if point_cloud_seq.ndim == 2:
        return mask_green_blue_points_for_policy(
            point_cloud_seq, min_keep_points=min_keep_points, remove_mode=remove_mode
        )
    elif point_cloud_seq.ndim == 3:
        return np.stack([
            mask_green_blue_points_for_policy(
                point_cloud_seq[t], min_keep_points=min_keep_points, remove_mode=remove_mode
            ) for t in range(point_cloud_seq.shape[0])
        ], axis=0)
    elif point_cloud_seq.ndim == 4:
        return np.stack([
            np.stack([
                mask_green_blue_points_for_policy(
                    point_cloud_seq[b, t], min_keep_points=min_keep_points, remove_mode=remove_mode
                ) for t in range(point_cloud_seq.shape[1])
            ], axis=0) for b in range(point_cloud_seq.shape[0])
        ], axis=0)
    else:
        raise ValueError(f"Unsupported point cloud shape: {point_cloud_seq.shape}")


def mask_green_blue_points_zero(point_cloud: np.ndarray, remove_mode: str = 'gb'):
    out = point_cloud.copy()

    if remove_mode == 'none':
        return out

    is_red, is_green, is_blue, is_neutral = classify_color_points(out)
    
    if remove_mode == 'gb':
        remove_mask = is_green | is_blue
    elif remove_mode == 'rb':
        remove_mask = is_red | is_blue
    elif remove_mode == 'b':
        remove_mask = is_blue
    elif remove_mode == 'r':
        remove_mask = is_red
    else:
        return out

    if remove_mask.sum() == 0:
        return out

    out[remove_mask, 0] = 10.0
    out[remove_mask, 1] = 10.0
    out[remove_mask, 2] = 10.0
    out[remove_mask, 3] = 0.0
    out[remove_mask, 4] = 0.0
    out[remove_mask, 5] = 0.0
    return out


def save_pointcloud_npy_and_png(
    point_cloud: np.ndarray, save_dir: str, prefix: str = "eval_pc",
    elev: float = 25, azim: float = 45, point_size: float = 4.0,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    npy_path = os.path.join(save_dir, f"{prefix}_{ts}.npy")
    png_path = os.path.join(save_dir, f"{prefix}_{ts}.png")

    np.save(npy_path, point_cloud)

    xyz = point_cloud[:, :3]
    rgb = _pc_rgb_to_01(point_cloud[:, 3:6])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=point_size, depthshade=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 + 1e-6

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    return npy_path, png_path


def save_pointcloud_compare_png(
    raw_point_cloud: np.ndarray, masked_point_cloud: np.ndarray,
    save_dir: str, prefix: str = "eval_pc_compare",
    elev: float = 25, azim: float = 45, point_size: float = 4.0,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(save_dir, f"{prefix}_{ts}.png")

    raw_xyz = raw_point_cloud[:, :3]
    raw_rgb = _pc_rgb_to_01(raw_point_cloud[:, 3:6])

    masked_xyz = masked_point_cloud[:, :3]
    masked_rgb = _pc_rgb_to_01(masked_point_cloud[:, 3:6])

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2], c=raw_rgb, s=point_size, depthshade=False)
    ax1.set_title("Raw Point Cloud")
    ax1.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(masked_xyz[:, 0], masked_xyz[:, 1], masked_xyz[:, 2], c=masked_rgb, s=point_size, depthshade=False)
    ax2.set_title("Masked Point Cloud (colors removed)")
    ax2.view_init(elev=elev, azim=azim)

    xyz = raw_xyz
    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 + 1e-6

    for ax in [ax1, ax2]:
        ax.set_xlim(x_mid - radius, x_mid + radius)
        ax.set_ylim(y_mid - radius, y_mid + radius)
        ax.set_zlim(z_mid - radius, z_mid + radius)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path


class RobotRunner(BaseRunner):

    def __init__(
        self,
        output_dir=None,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        task_name=None
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.obs = deque(maxlen=n_obs_steps + 1)
        self.env = None

        self.dump_eval_pointcloud = True
        self.dump_eval_pointcloud_once = True
        self.dump_eval_pointcloud_dir = os.path.join(
            output_dir if output_dir is not None else ".",
            "debug_pointcloud"
        )
        self._has_dumped_pointcloud = False
        self.dump_masked_pointcloud_compare = True

        self.intervene_pointcloud_for_policy = True
        self.intervene_task_names = None

        self.step_count = 0
        
        # 状态机管理
        self.phase = 0
        self.red_was_in_air = False
        
        # 放置时长防卡死计数器
        self.red_placed_count = 0
        self.green_placed_count = 0

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()
        self._has_dumped_pointcloud = False
        self.step_count = 0
        
        self.phase = 0
        self.red_was_in_air = False
        self.red_placed_count = 0
        self.green_placed_count = 0

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy: BasePolicy, observaton=None) -> np.ndarray:
        self.step_count += 1
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)
        obs = self.get_n_steps_obs()

        # ========== 点云状态机切换逻辑 ==========
        if "point_cloud" in obs:
            pc_raw = obs["point_cloud"]
            if isinstance(pc_raw, torch.Tensor):
                pc_raw = pc_raw.detach().cpu().numpy()
            
            if pc_raw.ndim == 3:
                pc_latest = pc_raw[-1]
            elif pc_raw.ndim == 2:
                pc_latest = pc_raw
            elif pc_raw.ndim == 4:
                pc_latest = pc_raw[0, -1]
            else:
                pc_latest = pc_raw

            if self.phase == 0:
                is_red, _, _, _ = classify_color_points(pc_latest)
                if not self.red_was_in_air:
                    if not check_red_on_bottom(pc_latest):
                        self.red_was_in_air = True
                else:
                    is_on_bottom = check_red_on_bottom(pc_latest)
                    is_released = check_robot_away_from_block(pc_latest, is_red, safe_dist=0.035)
                    
                    if is_on_bottom:
                        self.red_placed_count += 1
                    else:
                        self.red_placed_count = 0
                        
                    # 只要方块平稳放下，且（夹爪退开 3.5cm 或 稳放超过10帧防卡死兜底），立刻切！
                    if is_on_bottom and (is_released or self.red_placed_count >= 10):
                        self.phase = 1
                        reason = "released" if is_released else "timeout fallback"
                        print(f"[Phase Switch] Step {self.step_count}: Red placed & {reason}. Switch to Phase 1.")
                        
            elif self.phase == 1:
                _, is_green, _, _ = classify_color_points(pc_latest)
                
                is_stacked = check_green_on_red(pc_latest)
                is_released = check_robot_away_from_block(pc_latest, is_green, safe_dist=0.035)
                
                if is_stacked:
                    self.green_placed_count += 1
                else:
                    self.green_placed_count = 0
                    
                if is_stacked and (is_released or self.green_placed_count >= 10):
                    self.phase = 2
                    reason = "released" if is_released else "timeout fallback"
                    print(f"[Phase Switch] Step {self.step_count}: Green stacked & {reason}. Switch to Phase 2.")

        # ========== 根据状态决定屏蔽模式 ==========
        if self.phase == 0:
            remove_mode = 'gb'    # phase 0：一开始只保留红
        elif self.phase == 1:
            # 💡【核心修复】：必须让策略看见红方块（目标底座），否则绿方块不知道叠在哪，导致滑落。
            # 所以这里必须用 'b'（只屏蔽蓝色），绝不能连红色也屏蔽。
            remove_mode = 'b'     
        else:
            remove_mode = 'none'  # phase 2：绿叠到红上，恢复所有方块可视性

        # ===== Dump raw and masked point cloud for debug =====
        if self.dump_eval_pointcloud and "point_cloud" in obs:
            should_dump = True
            if self.dump_eval_pointcloud_once and self._has_dumped_pointcloud:
                should_dump = False

            if should_dump:
                pc = obs["point_cloud"]

                if pc.ndim == 3:
                    pc_latest = pc[-1].copy()
                    
                    save_pointcloud_npy_and_png(
                        pc_latest,
                        save_dir=self.dump_eval_pointcloud_dir,
                        prefix=f"{self.task_name}_raw_latest"
                    )

                    seq_path = os.path.join(
                        self.dump_eval_pointcloud_dir,
                        f"{self.task_name}_raw_seq_{time.strftime('%Y%m%d_%H%M%S')}.npy"
                    )
                    np.save(seq_path, pc)

                    if self.dump_masked_pointcloud_compare:
                        pc_masked_vis = mask_green_blue_points_zero(pc_latest, remove_mode=remove_mode)
                        
                        save_pointcloud_npy_and_png(
                            pc_masked_vis,
                            save_dir=self.dump_eval_pointcloud_dir,
                            prefix=f"{self.task_name}_masked_vis_latest"
                        )

                        save_pointcloud_compare_png(
                            pc_latest,
                            pc_masked_vis,
                            save_dir=self.dump_eval_pointcloud_dir,
                            prefix=f"{self.task_name}_raw_vs_masked_vis"
                        )

                elif pc.ndim == 2:
                    pc_single = pc.copy()
                    
                    save_pointcloud_npy_and_png(
                        pc_single,
                        save_dir=self.dump_eval_pointcloud_dir,
                        prefix=f"{self.task_name}_raw_single"
                    )

                    if self.dump_masked_pointcloud_compare:
                        pc_masked_vis = mask_green_blue_points_zero(pc_single, remove_mode=remove_mode)
                        
                        save_pointcloud_npy_and_png(
                            pc_masked_vis,
                            save_dir=self.dump_eval_pointcloud_dir,
                            prefix=f"{self.task_name}_masked_vis_single"
                        )

                        save_pointcloud_compare_png(
                            pc_single,
                            pc_masked_vis,
                            save_dir=self.dump_eval_pointcloud_dir,
                            prefix=f"{self.task_name}_raw_vs_masked_vis"
                        )
                self._has_dumped_pointcloud = True

        # ===== Real intervention: modify the point cloud actually fed into policy =====
        should_intervene = self.intervene_pointcloud_for_policy and "point_cloud" in obs
        if should_intervene and self.intervene_task_names is not None:
            should_intervene = (self.task_name in self.intervene_task_names)

        if should_intervene:
            obs["point_cloud"] = mask_green_blue_sequence_for_policy(
                obs["point_cloud"],
                min_keep_points=64,
                remove_mode=remove_mode,
            )

        # create obs dict
        np_obs_dict = dict(obs)

        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

        # run policy
        with torch.no_grad():
            obs_dict_input = {}
            obs_dict_input["point_cloud"] = obs_dict["point_cloud"].unsqueeze(0)
            obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

            action_dict = policy.predict_action(obs_dict_input)

        # device transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)
        return action

    def run(self, policy: BasePolicy):
        pass


if __name__ == "__main__":
    test = RobotRunner("./")
    print("ready")