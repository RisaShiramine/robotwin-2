#!/bin/bash
# RoboTwin Evaluation Script for stack_block_one task
# Usage: bash eval_stack_block_one.sh [checkpoint_id] [seed] [gpu_id]

policy_name=RDT
task_name=stack_block_one
task_config=stack_block_one_eval
model_name=rdt-finetune-robotwin-170m-oneblock

# Default values
checkpoint_id=${1:-10000}  # Default: latest checkpoint 10000
seed=${2:-0}               # Default seed: 0
gpu_id=${3:-0}             # Default GPU: 0

DEBUG=False
export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "=========================================="
echo "🤖 RDT Evaluation on RoboTwin"
echo "=========================================="
echo "Task Name:       ${task_name}"
echo "Task Config:     ${task_config}"
echo "Model Name:      ${model_name}"
echo "Checkpoint ID:   ${checkpoint_id}"
echo "Seed:            ${seed}"
echo "GPU ID:          ${gpu_id}"
echo "=========================================="

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --checkpoint_id ${checkpoint_id} \
    --policy_name ${policy_name}

echo ""
echo "✅ Evaluation complete!"
echo "📹 Videos saved to: eval_result/${task_name}/${policy_name}/${task_config}/${model_name}/"
