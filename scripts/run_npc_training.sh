#!/bin/bash
# ============================================================================
# NPC对话模型 - Agentic RL训练脚本（单轮）
#
# 设计要点：
#   - GRPO + 非对称clip (low=0.2, high=0.28) + 小KL约束
#   - 每轮训练 1 epoch（数据飞轮负责课程递增，超参保持稳定）
#   - 训练后自动merge checkpoint
#   - 超参数各轮一致，只变数据和模型初始化
#
# 用法:
#   bash run_npc_training.sh              # 默认 8xH200
#   bash run_npc_training.sh a100         # 4xA100
#   bash run_npc_training.sh 4090         # 8x4090
#
# 环境变量覆盖:
#   MODEL_PATH=./checkpoints/round1/merge bash run_npc_training.sh
#   DATA_PATH=./data/round2_train.parquet bash run_npc_training.sh
#   NPC_ABLATION=clip_high NPC_CONFIG_ONLY=1 bash run_npc_training.sh h200
# ============================================================================

set -ex
ulimit -n 65535 || true

# ====== 路径配置 ======
export PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-${PROJECT_ROOT}/data/val.parquet}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints}"
export TOOL_CONFIG="${PROJECT_ROOT}/rl/tool_config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ====== 根据参数选择硬件Profile ======
HW_PROFILE="${1:-h200}"

case "${HW_PROFILE}" in
  a100|A100|4xa100)
    N_GPUS=4
    NNODES=1
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=8
    PPO_MICRO_BATCH_SIZE=4
    ROLLOUT_N=8
    MAX_PROMPT_LENGTH=3072
    MAX_RESPONSE_LENGTH=2048
    MAX_TURNS=8
    LR_DEFAULT=5e-7
    GPU_MEMORY_UTIL=0.85
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    GRADIENT_CHECKPOINTING=True
    ROLLOUT_TP=1
    LOG_PROB_MICRO_BS=16
    EXPERIMENT_NAME="qwen3_8b_npc_4xA100"
    ;;

  4090|RTX4090|8x4090)
    N_GPUS=8
    NNODES=1
    TRAIN_BATCH_SIZE=16
    PPO_MINI_BATCH_SIZE=4
    PPO_MICRO_BATCH_SIZE=2
    ROLLOUT_N=4
    MAX_PROMPT_LENGTH=2048
    MAX_RESPONSE_LENGTH=1536
    MAX_TURNS=8
    LR_DEFAULT=5e-7
    GPU_MEMORY_UTIL=0.92
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    GRADIENT_CHECKPOINTING=True
    ROLLOUT_TP=2
    LOG_PROB_MICRO_BS=4
    EXPERIMENT_NAME="qwen3_8b_npc_8x4090"
    ;;

  h200|H200|8xh200|*)
    N_GPUS=8
    NNODES=1
    TRAIN_BATCH_SIZE=64
    PPO_MINI_BATCH_SIZE=16
    PPO_MICRO_BATCH_SIZE=8
    ROLLOUT_N=16
    MAX_PROMPT_LENGTH=4096
    MAX_RESPONSE_LENGTH=2048
    MAX_TURNS=8
    LR_DEFAULT=5e-7
    GPU_MEMORY_UTIL=0.90
    PARAM_OFFLOAD=False
    OPTIMIZER_OFFLOAD=False
    GRADIENT_CHECKPOINTING=True
    ROLLOUT_TP=1
    LOG_PROB_MICRO_BS=16
    EXPERIMENT_NAME="qwen3_8b_npc_8xH200"
    ;;
esac
ROLLOUT_N="${NPC_ROLLOUT_N:-${ROLLOUT_N}}"

# ====== 算法选择 ======
# 本入口只承诺GRPO。GSPO需要序列级ratio/clip实现，不能只替换adv_estimator。
RL_VARIANT="${RL_VARIANT:-${RL_ALGORITHM:-grpo}}"
NPC_ABLATION="${NPC_ABLATION:-baseline}"
export RL_VARIANT
export NPC_ABLATION
if [ -z "${OUTPUT_DIR_WAS_SET}" ]; then
  export OUTPUT_DIR="${OUTPUT_DIR}/${RL_VARIANT}_${NPC_ABLATION}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RL_VARIANT}_${NPC_ABLATION}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_rl_variant.py" "${RL_VARIANT}" \
  --ablation "${NPC_ABLATION}" --max-response-length "${MAX_RESPONSE_LENGTH}"
mapfile -t EFFECTIVE_ALGO_OVERRIDES < <(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_rl_variant.py" "${RL_VARIANT}" \
    --ablation "${NPC_ABLATION}" --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --print-effective-overrides
)

# ====== 固定超参（各轮一致，由数据飞轮驱动难度递增） ======
CLIP_RATIO_C=10.0
LR="${LR:-${LR_DEFAULT}}"
KL_COEFF="${KL_COEFF:-0.01}"  # 中心值；需按目标veRL的token归一化语义调参
ENTROPY_COEFF=0
TOTAL_EPOCHS=1           # 每轮1 epoch，避免在同质数据上过拟合
SAVE_FREQ=6
TEST_FREQ=3

# ====== 打印配置 ======
echo "============================================================"
echo "NPC Dialogue Agentic RL Training (${RL_VARIANT^^})"
echo "============================================================"
echo "Hardware Profile : ${HW_PROFILE}"
echo "Algorithm        : ${RL_VARIANT}"
echo "Ablation         : ${NPC_ABLATION}"
echo "GPUs             : ${N_GPUS}"
echo "Model            : ${MODEL_PATH}"
echo "Data             : ${DATA_PATH}"
echo "Batch Size       : ${TRAIN_BATCH_SIZE}"
echo "Rollout N        : ${ROLLOUT_N}"
echo "Max Turns        : ${MAX_TURNS}"
echo "LR               : ${LR}"
echo "KL Coeff         : ${KL_COEFF}"
echo "Config contract  : printed by validate_rl_variant.py above"
echo "Epochs           : ${TOTAL_EPOCHS}"
echo "Output           : ${OUTPUT_DIR}"
echo "============================================================"
if [ "${NPC_CONFIG_ONLY:-0}" = "1" ]; then
    echo "Config-only preflight complete; training was not started."
    exit 0
fi

# ====== 启动训练 ======
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/verl_entrypoint.py" verl.trainer.main_ppo \
    "${EFFECTIVE_ALGO_OVERRIDES[@]}" \
    \
    data.train_files="${DATA_PATH}" \
    data.val_files="${VAL_DATA_PATH}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEFF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    \
    custom_reward_function.path="${PROJECT_ROOT}/rl/npc_reward_function.py" \
    custom_reward_function.name=compute_score_npc_dialogue \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name="npc_dialogue_rl" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger=['console'] \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.critic_warmup=0

# ====== 训练后自动Merge Checkpoint ======
echo "============================================================"
echo "Training complete. Merging checkpoint..."
echo "============================================================"

LATEST_STEP=$(cat ${OUTPUT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -z "$LATEST_STEP" ]; then
    echo "Warning: Could not find latest_checkpointed_iteration.txt, skipping merge"
else
    echo "Merging checkpoint from global_step_${LATEST_STEP}"
    "${PYTHON_BIN}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${OUTPUT_DIR}/global_step_${LATEST_STEP}/actor \
        --target_dir ${OUTPUT_DIR}/merge

    echo "Merged model saved to: ${OUTPUT_DIR}/merge"
    echo "Use this as MODEL_PATH for next round:"
    echo "  MODEL_PATH=${OUTPUT_DIR}/merge bash run_npc_training.sh ${HW_PROFILE}"
fi
