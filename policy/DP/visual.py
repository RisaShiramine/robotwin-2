import zarr
import numpy as np
import matplotlib.pyplot as plt

def visualize_episode_actions(zarr_path, episode_idx=0):
    # 1. 打开 Zarr 数据集
    print(f"Loading dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # 2. 读取动作和 episode 边界信息
    actions = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]
    
    # 3. 计算目标 episode 的起始和结束索引
    if episode_idx >= len(episode_ends):
        raise ValueError(f"Episode {episode_idx} out of bounds. Max is {len(episode_ends)-1}.")
        
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    
    # 提取当前 episode 的动作数据, shape: (episode_length, 14)
    ep_actions = actions[start_idx:end_idx]
    episode_length = ep_actions.shape[0]
    print(f"Visualizing Episode {episode_idx} | Length: {episode_length} steps")

    # 4. 开始绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_steps = np.arange(episode_length)

    # 上半部分：绘制前 7 维 (通常是左臂/Arm 1)
    for i in range(7):
        axes[0].plot(time_steps, ep_actions[:, i], label=f'Dim {i}')
    axes[0].set_title(f'Episode {episode_idx} - Action Dimensions 0-6 (e.g., Arm 1)')
    axes[0].set_ylabel('Action Value')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 下半部分：绘制后 7 维 (通常是右臂/Arm 2)
    for i in range(7, 14):
        axes[1].plot(time_steps, ep_actions[:, i], label=f'Dim {i}')
    axes[1].set_title(f'Episode {episode_idx} - Action Dimensions 7-13 (e.g., Arm 2)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Action Value')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 调整布局以防止图例被裁切
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为你实际的路径
    dataset_path = '/mnt/hdd/Project/RoboTwin/policy/DP/data/stack_block_one-demo_clean-50.zarr'
    
    # 可视化第 0 个 episode。你可以改成 1, 2, 3 等查看其他片段
    visualize_episode_actions(dataset_path, episode_idx=1)