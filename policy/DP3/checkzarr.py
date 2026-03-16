import zarr
import numpy as np
import matplotlib.pyplot as plt

def visualize_frame(zarr_path, frame_idx=0):
    root = zarr.open(zarr_path, mode='r')
    points = root['data/point_cloud'][frame_idx] # 取一帧 (1024, 6)
    
    xyz = points[:, :3]
    color_info = points[:, 3:]

    # 如果数值在 [0, 255]，归一化到 [0, 1] 以便 matplotlib 显示
    if np.max(color_info) > 1.1:
        color_info = color_info / 255.0
    
    # 强制裁剪到 [0, 1] 防止异常值
    color_info = np.clip(color_info, 0, 1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用后三维作为颜色 c
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color_info, s=10)
    
    ax.set_title(f"Frame {frame_idx} - Color Check")
    plt.show()

visualize_frame("data/stack_green_block-demo_clean-50.zarr", frame_idx=100)