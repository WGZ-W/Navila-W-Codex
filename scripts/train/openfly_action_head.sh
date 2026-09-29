#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

OPENFLY_RLDS_ROOT="${OPENFLY_RLDS_ROOT:-/mnt/sdc/weiguanzhao/OpenFly-rlds-my}"
export OPENFLY_RLDS_ROOT
if [[ ! -d "${OPENFLY_RLDS_ROOT}/vln_history" ]]; then
    echo "Missing ${OPENFLY_RLDS_ROOT}/vln_history" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-a8cheng/navila-llama3-8b-8f}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/navila-openfly-action-head}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --model_name_or_path "${MODEL_PATH}" \
    --version llama_3 \
    --data_mixture openfly \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --num_video_frames 8 \
    --enable_action_head True \
    --enable_history_mamba True \
    --history_num_frames 4 \
    --num_actions 10 \
    --action_head_dropout 0.1 \
    --action_loss_weight 1.0 \
    --tune_action_head True \
    --tune_history_mamba True \
    --tune_history_projector True \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --dispatch_batches False \
    --ddp_find_unused_parameters False \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none
