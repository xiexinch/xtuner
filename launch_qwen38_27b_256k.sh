#!/usr/bin/env bash
set -euo pipefail
set -x

# Qwen3.8-27B native 256K SFT.
# Default topology: 32 GPUs = 4 nodes x 8 GPUs, SP=4, DP=8, TP=EP=1.
#
# Every path can be overridden without editing this file, for example:
#   MODEL_PATH=/path/to/Qwen3.8-27B CONFIG_FILE=/path/to/sft_qwen38_27b.py bash "$0"

gpu_group=${GPU_GROUP:-llmagent_gpu}
namespace=${NAMESPACE:-ailab-llmagent}
gpus_per_node=${GPUS_PER_NODE:-8}
num_nodes=${NUM_NODES:-4}

xtuner_path=${XTUNER_PATH:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/xtuner}
config_file=${CONFIG_FILE:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/configs/sft_qwen38_27b.py}

# Override MODEL_PATH with the actual downloaded snapshot path before launch.
model_path=${MODEL_PATH:-/mnt/shared-storage-user/songdemin/user/haijun/code/gitlab/claude_code/models/mmodels--Qwen--Qwen3.8-27B/snapshots/REPLACE_WITH_SNAPSHOT}
meta_data_path=${META_DATA_PATH:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/data_recipes/frontend-0807.json}
output_root=${OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/outputs/qwen38-27b-256k_frontend-0807}

petrel_sdk_path=${PETREL_SDK_PATH:-/mnt/shared-storage-user/xiexinchen/petrel_oss_sdk-2.3.24}
tokenizer_cache_dir=${TOKENIZER_CACHE_DIR:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/tokenizer_cache}
ceph_config=${CEPH_CONFIG:-}
log_dir=${LOG_DIR:-/mnt/shared-storage-gpfs2/intern-pretrain-shared02/xiexinchen/xtuner_scripts/logs}
enable_fp8=${ENABLE_FP8:-false}
glm52_data_compat=${GLM52_DATA_COMPAT:-false}

image=${IMAGE:-registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47}
meta_name=$(basename "${meta_data_path}")
meta_name=${meta_name%.*}
run_tag=${RUN_TAG:-${meta_name}-256k-$(date +%Y%m%d-%H%M)}
job_name=${JOB_NAME:-qwen38-27b-256k-${run_tag}}
work_dir=${WORK_DIR:-${output_root}/${run_tag}}
# `%r` is replaced with TORCHRUN_NODE_RANK inside each worker, so every node
# gets a separate file while the name remains derived entirely from job_name.
log_file=${LOG_FILE:-${log_dir}/${job_name}_node%r.log}

rjob submit \
    --name="${job_name}" \
    --task_name t0 \
    --gpu="${gpus_per_node}" --memory=524288 --cpu=64 \
    --charged-group="${gpu_group}" \
    --namespace "${namespace}" \
    --private-machine=group \
    -P "${num_nodes}" \
    --image "${image}" \
    --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
    --mount=gpfs://gpfs2/intern-agent-delivery:/mnt/shared-storage-gpfs2/intern-agent-delivery \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --mount=gpfs://gpfs1/xiexinchen:/mnt/shared-storage-user/xiexinchen \
    --mount=gpfs://gpfs1/llm-dev:/mnt/shared-storage-user/llm-dev \
    --mount=gpfs://gpfs1/puyudelivery:/mnt/shared-storage-user/puyudelivery \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --host-network=true \
    --gang-start=true \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --custom-resources brainpp.cn/fuse=1 \
    -e DISTRIBUTED_JOB=true \
    -e XTUNER_PATH="${xtuner_path}" \
    -e CONFIG_FILE="${config_file}" \
    -e MODEL_PATH="${model_path}" \
    -e META_DATA_PATH="${meta_data_path}" \
    -e CEPH_CONFIG="${ceph_config}" \
    -e WORK_DIR="${work_dir}" \
    -e TOKENIZER_CACHE_DIR="${tokenizer_cache_dir}" \
    -e PETREL_SDK_PATH="${petrel_sdk_path}" \
    -e LOG_DIR="${log_dir}" \
    -e LOG_FILE="${log_file}" \
    -e GPUS_PER_NODE="${gpus_per_node}" \
    -e TORCHRUN_NNODES="${num_nodes}" \
    -e CHAT_TEMPLATE_NAME=qwen3.8-vl \
    -e ENABLE_THINKING=true \
    -e REASONING_EFFORT=xhigh \
    -e PRESERVE_THINKING=true \
    -e GLM52_DATA_COMPAT="${glm52_data_compat}" \
    -e SAMPLE_MAX_LENGTH="${SAMPLE_MAX_LENGTH:-262144}" \
    -e PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-262144}" \
    -e MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-262144}" \
    -e GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" \
    -e SP_SIZE="${SP_SIZE:-4}" \
    -e TP_SIZE=1 \
    -e EP_SIZE=1 \
    -e NUM_WORKERS="${NUM_WORKERS:-4}" \
    -e PACK_EXTRA_BUFFER_SIZE="${PACK_EXTRA_BUFFER_SIZE:-20}" \
    -e RAND_VIDEO_MAX_FRAMES="${RAND_VIDEO_MAX_FRAMES:-24}" \
    -e MAX_PIXELS="${MAX_PIXELS:-16777216}" \
    -e TOKENIZE_DEBUG=false \
    -e LR=2e-5 \
    -e LR_MIN=1e-6 \
    -e WEIGHT_DECAY=0.05 \
    -e WARMUP_RATIO=0.1 \
    -e RECOMPUTE_RATIO=1.0 \
    -e VISION_RECOMPUTE_RATIO=1.0 \
    -e LOSS_REDUCTION=square \
    -e LOSS_CHUNK_SIZE=1024 \
    -e TORCH_COMPILE="${TORCH_COMPILE:-true}" \
    -e ENABLE_FP8="${enable_fp8}" \
    -e TOTAL_STEP="${TOTAL_STEP:-}" \
    -e TOTAL_EPOCH="${TOTAL_EPOCH:-2}" \
    -e DEBUG_SKIP_SAVE="${DEBUG_SKIP_SAVE:-false}" \
    -e HF_INTERVAL=500 \
    -e HF_MAX_KEEP=2 \
    -e CHECKPOINT_INTERVAL=500 \
    -e CHECKPOINT_MAXKEEP=2 \
    -- bash -lc '
        set -euo pipefail
        set -x

        export PYTHONPATH="${PETREL_SDK_PATH}:${XTUNER_PATH}:${PYTHONPATH:-}"
        export TORCHRUN_NODE_RANK="${NODE_RANK:-${RANK:-}}"
        export MASTER_PORT="${MASTER_PORT:-29500}"
        export TOKENIZERS_PARALLELISM=false
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

        if [ -z "${TORCHRUN_NODE_RANK}" ] || [ -z "${MASTER_ADDR:-}" ]; then
            echo "ERROR: require NODE_RANK/RANK and MASTER_ADDR for multi-node torchrun."
            env | sort | grep -E "^(NODE_RANK|RANK|NODE_COUNT|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|HOSTNAME)=" || true
            exit 2
        fi

        if [ "${TORCHRUN_NNODES}" != "1" ] && { [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
            echo "ERROR: MASTER_ADDR=${MASTER_ADDR} is invalid for ${TORCHRUN_NNODES}-node torchrun."
            exit 2
        fi

        node_log_file="${LOG_FILE//%r/${TORCHRUN_NODE_RANK}}"
        mkdir -p "$(dirname "${node_log_file}")"
        exec > >(tee "${node_log_file}") 2>&1

        cd "${XTUNER_PATH}"
        echo "TORCHRUN_NNODES=${TORCHRUN_NNODES}"
        echo "TORCHRUN_NODE_RANK=${TORCHRUN_NODE_RANK}"
        echo "MASTER_ADDR=${MASTER_ADDR}"
        echo "MASTER_PORT=${MASTER_PORT}"
        echo "CONFIG_FILE=${CONFIG_FILE}"
        echo "MODEL_PATH=${MODEL_PATH}"
        echo "META_DATA_PATH=${META_DATA_PATH}"
        echo "WORK_DIR=${WORK_DIR}"
        echo "LOG_FILE=${node_log_file}"
        test -f "${CONFIG_FILE}"
        test -f "${META_DATA_PATH}"
        test -f "${MODEL_PATH}/config.json"
        python -c "import xtuner; print(\"xtuner import:\", xtuner.__file__)"

        torchrun \
            --nproc-per-node="${GPUS_PER_NODE}" \
            --nnodes="${TORCHRUN_NNODES}" \
            --node_rank="${TORCHRUN_NODE_RANK}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            xtuner/v1/train/cli/sft.py \
            --config "${CONFIG_FILE}"
    '
