import os
import shutil
import argparse
import numpy as np
import zarr
import h5py


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at:\n{dataset_path}\n")

    with h5py.File(dataset_path, "r") as root:
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]

    return vector, pointcloud


def validate_pointcloud(pointcloud: np.ndarray):
    """
    Validate whether pointcloud is xyz or xyzrgb.
    Expected shapes:
      - (T, N, 3)
      - (T, N, 6)
    """
    if pointcloud.ndim != 3:
        raise ValueError(f"Expected pointcloud ndim=3, got shape={pointcloud.shape}")

    if pointcloud.shape[-1] not in (3, 6):
        raise ValueError(
            f"Expected pointcloud last dim to be 3 or 6, got shape={pointcloud.shape}"
        )

    if pointcloud.shape[-1] == 6:
        rgb = pointcloud[..., 3:6]
        rgb_min = float(rgb.min())
        rgb_max = float(rgb.max())
        print(f"[process_data] detected xyzrgb pointcloud. rgb range = [{rgb_min:.4f}, {rgb_max:.4f}]")
    else:
        print("[process_data] detected xyz-only pointcloud.")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin hdf5 episodes to DP3 zarr format.")
    parser.add_argument("task_name", type=str, help="The name of the task")
    parser.add_argument("task_config", type=str, help="Task config name, e.g. demo_clean")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []

    total_count = 0
    current_ep = 0

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        vector_all, pointcloud_all = load_hdf5(load_path)
        validate_pointcloud(pointcloud_all)

        T = vector_all.shape[0]
        if pointcloud_all.shape[0] != T:
            raise ValueError(
                f"State/action length and pointcloud length mismatch in {load_path}: "
                f"{T} vs {pointcloud_all.shape[0]}"
            )

        # Keep the original RoboTwin DP3 convention:
        #   obs at t uses state/pointcloud[t]
        #   action at t uses vector[t+1]
        for j in range(T):
            joint_state = vector_all[j]
            pointcloud = pointcloud_all[j]

            if j != T - 1:
                point_cloud_arrays.append(pointcloud.astype(np.float32))
                state_arrays.append(joint_state.astype(np.float32))
            if j != 0:
                joint_action_arrays.append(joint_state.astype(np.float32))

        total_count += T - 1
        episode_ends_arrays.append(total_count)
        current_ep += 1

    print()

    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    state_arrays = np.asarray(state_arrays, dtype=np.float32)
    point_cloud_arrays = np.asarray(point_cloud_arrays, dtype=np.float32)
    joint_action_arrays = np.asarray(joint_action_arrays, dtype=np.float32)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    print(f"[process_data] saved zarr to: {save_dir}")
    print(f"[process_data] point_cloud shape: {point_cloud_arrays.shape}")
    print(f"[process_data] state shape: {state_arrays.shape}")
    print(f"[process_data] action shape: {joint_action_arrays.shape}")


if __name__ == "__main__":
    main()
