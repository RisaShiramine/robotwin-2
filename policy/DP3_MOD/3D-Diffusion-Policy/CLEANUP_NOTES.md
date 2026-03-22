# Cleanup notes

Removed old experimental or duplicated files that were not needed for the new training path:

- `diffusion_policy_3d/dataset/planner_robot_dataset.py`
- `diffusion_policy_3d/env_runner/planner_controller.py`
- `diffusion_policy_3d/env_runner/robot_runner.py_exp`
- `diffusion_policy_3d/model/vision/pointnet_extractor.py.bak`
- `mask_eval_pointcloud.py`
- all `__pycache__` directories and `*.pyc`
- `diffusion_policy_3d.egg-info/`

The package now keeps a single training path:

`zarr dataset + optional timestep_labels.jsonl sidecar -> RobotDataset -> DP3`
