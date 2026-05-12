#!/bin/bash
# ============================================================================
# NPC对话模型RL训练脚本
# 支持三种硬件配置：
#   1. 8x H200 141GB (默认，推荐)
#   2. 4x A100 80GB
#   3. 8x RTX 4090 24GB
#
# 用法:
#   bash run_npc_training.sh              # 默认 8xH200
#   bash run_npc_training.sh a100         # 4xA100
#   bash run_npc_training.sh 4090         # 8x4090
# ============================================================================

set -ex
ulimit -n 65535

# ====== 路径配置 ======
export PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DATA_PATH="${PROJECT_ROOT}/data/train.parquet"
export VAL_DATA_PATH="${PROJECT_ROOT}/data/val.parquet"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export OUTPUT_DIR="${PROJECT_ROOT}/checkpoints"
export TOOL_CONFIG="${PROJECT_ROOT}/rl/tool_config.yaml"

# ====== 根据参数选择硬件Profile ======
HW_PROFILE="${1:-h200}"

case "${HW_PROFILE}" in
  # ─────────────────────────────────────────────────────────────────────────
  # Profile 1: 4x A100 80GB
  # 中等资源，param offload到CPU，TP=1（8B单卡可放）
  # 预计训练时间：~30h/epoch
  # ─────────────────────────────────────────────────────────────────────────
  a100|A100|4xa100)
    N_GPUS=4
    NNODES=1
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=8
    PPO_MICRO_BATCH_SIZE=4
    ROLLOUT_N=8
    MAX_PROMPT_LENGTH=3072
    MAX_RESPONSE_LENGTH=4096
    MAX_TURNS=10
    LR=5e-7
    GPU_MEMORY_UTIL=0.85
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    GRADIENT_CHECKPOINTING=True
    ROLLOUT_TP=1
    LOG_PROB_MICRO_BS=16
    EXPERIMENT_NAME="qwen3_8b_npc_4xA100"
    ;;

  # ─────────────────────────────────────────────────────────────────────────
  # Profile 2: 8x RTX 4090 24GB
  # 显存极为紧张（24GB/卡），必须激进优化：
  #   - TP=2（8B模型跨2卡，每卡~4B参数≈8GB）
  #   - 极小micro batch
  #   - 强制param+optimizer offload
  #   - 缩短序列长度
  #   - 减少rollout数量
  # 预计训练时间：~48-60h/epoch（受CPU offload瓶颈限制）
  # ─────────────────────────────────────────────────────────────────────────
  4090|RTX4090|8x4090)
    N_GPUS=8
    NNODES=1
    TRAIN_BATCH_SIZE=16           # 进一步减小（显存紧张）
    PPO_MINI_BATCH_SIZE=4         # 最小化
    PPO_MICRO_BATCH_SIZE=2        # 最小化（每步只处理2个样本）
    ROLLOUT_N=4                   # 最少可行rollout数（GRPO下界）
    MAX_PROMPT_LENGTH=2048        # 缩短prompt（NPC对话2K够用）
    MAX_RESPONSE_LENGTH=2048      # 缩短回复（强制简洁）
    MAX_TURNS=8                   # 减少对话轮数
    LR=3e-7                       # 更小LR（极小batch需要）
    GPU_MEMORY_UTIL=0.92          # 更激进的显存使用（4090无ECC）
    PARAM_OFFLOAD=True            # 必须开启
    OPTIMIZER_OFFLOAD=True        # 必须开启
    GRADIENT_CHECKPOINTING=True   # 必须开启
    ROLLOUT_TP=2                  # 8B模型需要TP=2才能在24GB卡上推理
    LOG_PROB_MICRO_BS=4           # 缩小避免OOM
    EXPERIMENT_NAME="qwen3_8b_npc_8x4090"
    ;;

  # ─────────────────────────────────────────────────────────────────────────
  # Profile 3: 8x H200 141GB
  # 资源充足，全力发挥：
  #   - 无需offload（141GB绰绰有余）
  #   - 大batch + 多rollout = 更低方差
  #   - 更长序列 = 更复杂对话
  #   - 可以直接对标论文8xH100配置
  # 预计训练时间：~12-16h/epoch
  # ─────────────────────────────────────────────────────────────────────────
  h200|H200|8xh200)
    N_GPUS=8
    NNODES=1
    TRAIN_BATCH_SIZE=64           # 论文原始配置
    PPO_MINI_BATCH_SIZE=16        # 论文原始配置
    PPO_MICRO_BATCH_SIZE=8        # 大micro batch（显存充足）
    ROLLOUT_N=16                  # 论文原始配置（更低方差）
    MAX_PROMPT_LENGTH=4096        # 更长prompt（支持复杂NPC策略）
    MAX_RESPONSE_LENGTH=6144      # 更长回复（支持复杂工具调用链）
    MAX_TURNS=12                  # 更多对话轮数
    LR=1e-6                       # 论文原始LR
    GPU_MEMORY_UTIL=0.80          # 保守一些（141GB不需要激进）
    PARAM_OFFLOAD=False           # 不需要offload
    OPTIMIZER_OFFLOAD=False       # 不需要offload
    GRADIENT_CHECKPOINTING=True   # 仍然开启（省显存给KV cache）
    ROLLOUT_TP=1                  # 8B模型单卡绰绰有余
    LOG_PROB_MICRO_BS=32          # 更大的log prob计算batch
    EXPERIMENT_NAME="qwen3_8b_npc_8xH200"
    ;;

  *)
    echo "Error: Unknown hardware profile '${HW_PROFILE}'"
    echo "Usage: $0 [h200|a100|4090]"
    echo ""
    echo "Profiles:"
    echo "  h200  - 8x H200 141GB (default, recommended)"
    echo "  a100  - 4x A100 80GB"
    echo "  4090  - 8x RTX 4090 24GB"
    exit 1
    ;;
esac

# ====== 算法选择 ======
# 支持: grpo (默认，已验证) | gspo (备选，序列级优化)
# 用法: RL_ALGORITHM=gspo bash run_npc_training.sh h200
RL_ALGORITHM="${RL_ALGORITHM:-grpo}"

# ====== 通用超参数 ======
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
KL_COEFF=0.001
ENTROPY_COEFF=0
WARMUP_STEPS=10
TOTAL_EPOCHS=3
SAVE_FREQ=50

# ====== 打印配置 ======
echo "============================================================"
echo "NPC Dialogue RL Training (verl + ${RL_ALGORITHM^^})"
echo "============================================================"
echo "Hardware Profile : ${HW_PROFILE}"
echo "Algorithm        : ${RL_ALGORITHM}"
echo "GPUs             : ${N_GPUS}"
echo "Model            : ${MODEL_PATH}"
echo "Batch Size       : ${TRAIN_BATCH_SIZE}"
echo "Rollout N        : ${ROLLOUT_N}"
echo "Max Prompt       : ${MAX_PROMPT_LENGTH}"
echo "Max Response     : ${MAX_RESPONSE_LENGTH}"
echo "Max Turns        : ${MAX_TURNS}"
echo "Learning Rate    : ${LR}"
echo "Param Offload    : ${PARAM_OFFLOAD}"
echo "Optimizer Offload: ${OPTIMIZER_OFFLOAD}"
echo "Rollout TP       : ${ROLLOUT_TP}"
echo "Output           : ${OUTPUT_DIR}"
echo "============================================================"

# ====== 启动训练 ======
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${RL_ALGORITHM} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=${KL_COEFF} \
    \
    data.train_files="${DATA_PATH}" \
    data.val_files="${VAL_DATA_PATH}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEFF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
    actor_rollout_ref.rollout.multi_turn.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    \
    reward_model.reward_manager=dapo \
    custom_reward_function.path="${PROJECT_ROOT}/rl/npc_reward_function.py" \
    custom_reward_function.name=compute_score_npc_dialogue \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.project_name="npc_dialogue_rl" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="wandb" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.warmup_steps=${WARMUP_STEPS} \
    trainer.critic_warmup=0

echo "============================================================"
echo "Training complete!"
echo "Profile    : ${HW_PROFILE}"
echo "Checkpoints: ${OUTPUT_DIR}"
echo "============================================================"
