# Planner sidecar training

This cleaned package removes the previous ad-hoc planner-token zarr path and supports a simpler training path:

- keep the original zarr dataset unchanged
- provide `timestep_labels.jsonl` as a sidecar file
- train DP3 with optional planner conditions from that sidecar

## What changed

- `diffusion_policy_3d/dataset/robot_dataset.py`
  - added optional `planner_labels_jsonl`
  - validates `flat_index`, `episode_id`, and `timestep` alignment against the zarr replay buffer
  - encodes planner fields into model-ready ids and anchor tensors
- `diffusion_policy_3d/policy/dp3.py`
  - adds planner-condition embeddings for:
    - stage
    - phase
    - active arm
    - source object
    - support kind
  - adds anchor MLP for source/support xyz
  - optional soft point weighting around source/support anchors
- `train_dp3.py`
  - automatically inspects `planner_labels_jsonl` and sets planner vocab sizes before model construction
- `diffusion_policy_3d/model/vision/pointnet_extractor.py`
  - supports optional per-point weights before max pooling

## Required sidecar fields

Each line in `timestep_labels.jsonl` should provide at least:

- `flat_index`
- `episode_id`
- `timestep`
- `stage_name`
- `phase_name`
- `active_arm`
- `source_object_id`
- `support_kind`
- `source_anchor_xyz`
- `support_anchor_xyz`
- `recommended_weights`

## Example config override

```bash
python train_dp3.py \
  task=demo_task \
  task.dataset.zarr_path=/abs/path/to/dataset.zarr \
  task.dataset.planner_labels_jsonl=/abs/path/to/timestep_labels.jsonl \
  policy.use_planner_condition=true \
  policy.planner_use_soft_point_weighting=true
```

If you want a minimal first baseline, keep `policy.planner_use_soft_point_weighting=false`.
