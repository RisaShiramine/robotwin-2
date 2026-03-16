import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_episode_actions(file_path):
    # 加载数据
    actions = np.load(file_path)
    steps = actions.shape[0]
    
    print(f"📦 正在分析: {file_path}")
    print(f"⏱️ 总步数: {steps}, 动作维度: {actions.shape[1]}")

    # 提取位置 (假设前3维是 x, y, z)
    pos = actions[:, :3]
    # 提取夹爪 (假设最后一维是 gripper)
    gripper = actions[:, -1]

    fig = plt.figure(figsize=(12, 5))

    # --- 子图 1: 3D 轨迹 ---
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=range(steps), cmap='viridis', s=10)
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.5)
    ax1.set_title("Robot EE Trajectory (3D)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    plt.colorbar(sc, label='Time Step', ax=ax1)

    # --- 子图 2: 夹爪状态 ---
    ax2 = fig.add_subplot(122)
    ax2.plot(range(steps), gripper, color='red', lw=2)
    ax2.set_title("Gripper State Over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Open/Close")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# 随便选一个 Episode 看看，比如第 0 个
plot_episode_actions('disassembled_actions/episode_000_actions.npy')