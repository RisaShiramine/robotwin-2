#!/bin/bash

policy_name=DP3_MOD
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}

planner_labels_jsonl=${7}
planner_eval_sequence_path=${8}
use_planner_condition=${9}
planner_debug=${10}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "use_planner_condition: ${use_planner_condition}"
echo "planner_labels_jsonl: ${planner_labels_jsonl}"
echo "planner_eval_sequence_path: ${planner_eval_sequence_path}"
echo "planner_debug: ${planner_debug}"

cd ../..

OVERRIDES=(
  --task_name "${task_name}"
  --task_config "${task_config}"
  --ckpt_setting "${ckpt_setting}"
  --expert_data_num "${expert_data_num}"
  --seed "${seed}"
  --policy_name "${policy_name}"
  --planner_labels_jsonl "${planner_labels_jsonl}"
  --use_planner_condition "${use_planner_condition}"
  --planner_debug "${planner_debug}"
)

if [ -n "${planner_eval_sequence_path}" ]; then
  OVERRIDES+=( --planner_eval_sequence_path "${planner_eval_sequence_path}" )
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
  --config "policy/${policy_name}/deploy_policy.yml" \
  --overrides "${OVERRIDES[@]}"
