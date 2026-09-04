#!/bin/bash
# NPC游戏推理Warm-up训练（Phase 1）
# 用简单的、有明确对错的游戏规则验证题训练工具调用能力
# 在正式Agentic RL之前执行

set -ex
ulimit -n 65535

# ====== 路径配置 ======
export PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DATA_PATH="${PROJECT_ROOT}/data/warmup_train.parquet"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/warmup"
export TOOL_CONFIG="${PROJECT_ROOT}/rl/tool_config.yaml"

# ====== 硬件配置 (默认8xH200 141GB) ======
N_GPUS=${N_GPUS:-8}
NNODES=1

# ====== Warm-up超参（比正式训练更激进，快速收敛） ======
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE=8
ROLLOUT_N=16
MAX_PROMPT_LENGTH=2048     # Warm-up的prompt更短
MAX_RESPONSE_LENGTH=2048   # Warm-up的回复更短（不需要复杂对话）
MAX_TURNS=4                # Warm-up只需要2-4轮（查询→判断→回复）
LR=1e-6                    # Warm-up可用更大LR（任务简单，收敛快）
TOTAL_EPOCHS=2             # 2个epoch足够学会工具调用语法
SAVE_FREQ=100

echo "=========================================="
echo "NPC Game Reasoning Warm-up (Phase 1)"
echo "GPUs: ${N_GPUS}x H200"
echo "Purpose: Learn tool-calling syntax + verify-before-act"
echo "=========================================="

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.005 \
    \
    data.train_files="${DATA_PATH}" \
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
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.gradient_checkpointing=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.88 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
    actor_rollout_ref.rollout.multi_turn.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${PROJECT_ROOT}/rl/warmup_reward_function.py" \
    custom_reward_function.name=compute_score_warmup \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.project_name="npc_dialogue_warmup" \
    trainer.experiment_name="qwen3_8b_game_reasoning" \
    trainer.logger="wandb" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.warmup_steps=5 \
    trainer.critic_warmup=0

echo "=========================================="
echo "Warm-up complete! Checkpoint: ${OUTPUT_DIR}"
echo "Next: run Phase 2 with --model_path=${OUTPUT_DIR}/latest"
echo "=========================================="
