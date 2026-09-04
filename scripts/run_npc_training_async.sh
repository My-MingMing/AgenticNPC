#!/bin/bash
# ============================================================================
# NPC对话模型 - Agentic RL训练脚本（全异步 / one-step-off-policy, 8xH200）
#
# 与 run_npc_training.sh 的区别：
#   同步版：rollout 和 train 共用同一批 GPU(hybrid engine),一个 step 内串行,
#           rollout 时训练卡空转、train 时推理引擎空转。
#   本版  ：训推解耦(hybrid_engine=False),GPU 拆成 Rollout 池 + Train 池两组常驻并行,
#           trainer 训练第 N 批时,rollout 已用刚同步的权重生成第 N+1 批 → 数据落后一步(off-policy)。
#           省的是"互相空等"的 wall-clock,不是计算量。本项目 rollout 受外部 LLM API 延迟主导,
#           训练能藏进 rollout 里,预计每步省 ~20-30%(取决于拆分与 API 并发)。
#
# ============================================================================
# 【verl 版本要求】必须 verl == 0.7.0     →  pip install verl==0.7.0
#   - one_step_off_policy(CUDA + SGLang 的全异步recipe)最后出现在 v0.7.0。
#     v0.7.1 起 verl 的 recipe/ 目录改为 git submodule(verl-project/verl-recipe),
#     该recipe被 async_flow 取代 —— async_flow 是 vLLM + Ascend NPU 向、不含 SGLang,
#     不适合本项目的 SGLang / CUDA / H200 技术栈。
#   - 且只有 v0.7.0 的 recipe/one_step_off_policy 带 agent_loop/ 目录(多轮 tool-use 必需);
#     v0.6.x 的同名recipe没有 agent_loop,跑不了本项目的多轮 agentic rollout。
#   → 因此锁定 verl v0.7.0(release tag,commit bec9ef7 之前的 0.7.0 tag)。
#   入口模块: python3 -m recipe.one_step_off_policy.main_ppo
#   默认 Hydra config: recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml
#
# 【本脚本只覆盖 H200】A100(4卡)/4090(8卡,rollout需TP=2)卡太少,拆分不划算,
#   继续用同步版 run_npc_training.sh 即可。
#
# 用法:
#   bash run_npc_training_async.sh
#   MODEL_PATH=./checkpoints/round1/merge bash run_npc_training_async.sh
#   该实验入口当前仅支持GRPO；GSPO需单独实现序列级ratio/clip后再开放。
# ============================================================================

set -ex
ulimit -n 65535

# ====== 路径配置 ======
export PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-${PROJECT_ROOT}/data/val.parquet}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints}"
export TOOL_CONFIG="${PROJECT_ROOT}/rl/tool_config.yaml"

# ====== GPU 拆分(核心)======
# 单机 8xH200。约定: n_gpus_rollout = Rollout(生成)池, 其余为 Train(训练)池。
#
#   >>> 你要的 5:3 拆分(Rollout=5, Train=3) <<<
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-5}
N_GPUS_TRAINING=$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))   # = 3
#
# ⚠️ 关于拆分方向(务必读):
#   verl 官方示例(GSM8K/math)都是 rollout少、train多(如 2:6、4:12),因为那类任务
#   生成是 GPU-bound,训练才是瓶颈。本项目相反:rollout 受外部 LLM API 延迟主导,
#   给 rollout 加 GPU 收益很小。因此 5:3 只有在"rollout 是本地 8B decode 瓶颈"时才最优;
#   若瓶颈其实是 mock_tool/mock_user/judge 的外部 API(很可能),应改成 3:5 甚至 2:6
#   (把更多卡给训练,让训练稳稳藏进长 rollout 内),并优先提升 rl/tool_config.yaml 的
#   rate_limit(当前=8)。改这里一行 N_GPUS_ROLLOUT 即可切换,建议两种都测一版取快者。

# ====== 生成/训练并行度 ======
# 8B 在 H200(141GB)上 TP=1 即可(单卡放得下),5 卡 rollout → 5 个 DP 副本。
# 注意:gen_tp 必须能整除 N_GPUS_ROLLOUT。5 只能被 1 或 5 整除,故 TP=1。
GEN_TP=${GEN_TP:-1}
# 训练池 3 卡:8B 的 FSDP2 全态(参数+梯度+Adam)约 128GB,分片到 3 卡每卡 ~43GB,H200 放得下,
# 因此可不做 offload。长序列(4K+10K=14K)靠 dynamic_bsz + gradient_checkpointing 扛。
FSDP_SIZE=${FSDP_SIZE:-3}
SP_SIZE=${SP_SIZE:-1}      # Ulysses 序列并行;3 卡下先关(=1),显存/OOM 再考虑开

# ====== 批量与长度(沿用 H200 profile)======
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
ROLLOUT_N=16
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=10000
MAX_TURNS=15
LR="${LR:-5e-7}"
GPU_MEMORY_UTIL=0.90
GRADIENT_CHECKPOINTING=True
LOG_PROB_MICRO_BS=16
EXPERIMENT_NAME="qwen3_8b_npc_8xH200_async_${N_GPUS_ROLLOUT}_${N_GPUS_TRAINING}"

# ====== 算法选择 ======
# 只支持GRPO。异步陈旧性通过版本上限、importance ratio监控和丢弃策略控制。
RL_VARIANT="${RL_VARIANT:-${RL_ALGORITHM:-grpo}}"
export RL_VARIANT
python3 "${PROJECT_ROOT}/scripts/validate_rl_variant.py" "${RL_VARIANT}" --max-response-length "${MAX_RESPONSE_LENGTH}"
case "${RL_VARIANT,,}" in
  grpo) ADV_ESTIMATOR=grpo; REWARD_MANAGER=naive; NORM_STD=True; CLIP_RATIO_LOW=0.20; CLIP_RATIO_HIGH=0.20; LOSS_AGG_MODE=sample-mean; USE_KL_REWARD=True; USE_KL_LOSS=True; DAPO_OVERRIDES=() ;;
  dapo) ADV_ESTIMATOR=grpo; REWARD_MANAGER=dapo; NORM_STD=False; CLIP_RATIO_LOW=0.20; CLIP_RATIO_HIGH=0.28; LOSS_AGG_MODE=token-mean; USE_KL_REWARD=False; USE_KL_LOSS=False; DAPO_OVERRIDES=(algorithm.filter_groups.enable=True algorithm.filter_groups.metric=seq_final_reward algorithm.filter_groups.max_num_gen_batches=4 reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} reward_model.reward_kwargs.overlong_buffer_cfg.enable=True reward_model.reward_kwargs.overlong_buffer_cfg.len=512 reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0) ;;
  ngrpo) ADV_ESTIMATOR=ngrpo; REWARD_MANAGER=naive; NORM_STD=True; CLIP_RATIO_LOW=0.20; CLIP_RATIO_HIGH=0.20; LOSS_AGG_MODE=sample-mean; USE_KL_REWARD=True; USE_KL_LOSS=True; DAPO_OVERRIDES=() ;;
  *) echo "Unsupported RL_VARIANT=${RL_VARIANT}" >&2; exit 2;;
esac

# ====== 固定超参(与同步版一致,便于对照)======
CLIP_RATIO_C=10.0
KL_COEFF="${KL_COEFF:-0.01}"
ENTROPY_COEFF=0
TOTAL_EPOCHS=1
SAVE_FREQ=6
TEST_FREQ=3

# ====== 打印配置 ======
echo "============================================================"
echo "NPC Dialogue Agentic RL — FULLY ASYNC (one-step-off-policy)"
echo "============================================================"
echo "verl required    : 0.7.0  (recipe.one_step_off_policy.main_ppo)"
echo "Algorithm        : ${RL_VARIANT}"
echo "GPU split        : Rollout=${N_GPUS_ROLLOUT}  Train=${N_GPUS_TRAINING}  (hybrid_engine=False)"
echo "Gen TP           : ${GEN_TP}   FSDP size: ${FSDP_SIZE}   SP: ${SP_SIZE}"
echo "Model            : ${MODEL_PATH}"
echo "Data             : ${DATA_PATH}"
echo "Batch / RolloutN : ${TRAIN_BATCH_SIZE} / ${ROLLOUT_N}"
echo "Max Turns        : ${MAX_TURNS}"
echo "Clip Ratio       : [${CLIP_RATIO_LOW}, ${CLIP_RATIO_HIGH}]  KL=${KL_COEFF}"
echo "Output           : ${OUTPUT_DIR}"
echo "============================================================"

# ====== 启动训练 ======
# 关键差异 vs 同步版:
#   actor_rollout_ref.hybrid_engine=False   → 训推解耦(全异步的开关)
#   trainer.n_gpus_per_node = 训练池        + rollout.nnodes / rollout.n_gpus_per_node = 生成池
#   actor.strategy=fsdp2 / critic.strategy=fsdp2  (one_step_off_policy 要求 fsdp2)
python3 "${PROJECT_ROOT}/scripts/verl_entrypoint.py" recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.norm_adv_by_std_in_grpo=${NORM_STD} \
    algorithm.use_kl_in_reward=${USE_KL_REWARD} \
    "${DAPO_OVERRIDES[@]}" \
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
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEFF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path="${PROJECT_ROOT}/rl/npc_reward_function.py" \
    custom_reward_function.name=compute_score_npc_dialogue \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.critic_warmup=0 \
    trainer.project_name="npc_dialogue_rl" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger=['console'] \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}"

# ====== 训练后自动 Merge Checkpoint(与同步版一致)======
echo "============================================================"
echo "Training complete. Merging checkpoint..."
echo "============================================================"

LATEST_STEP=$(cat ${OUTPUT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -z "$LATEST_STEP" ]; then
    echo "Warning: Could not find latest_checkpointed_iteration.txt, skipping merge"
else
    echo "Merging checkpoint from global_step_${LATEST_STEP}"
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${OUTPUT_DIR}/global_step_${LATEST_STEP}/actor \
        --target_dir ${OUTPUT_DIR}/merge
    echo "Merged model saved to: ${OUTPUT_DIR}/merge"
fi

# ============================================================================
# 上线前必做的冒烟验证(全异步 + 多轮 tool-use 是较新的组合,别直接开大规模):
#   1) 先确认 verl 版本:  python -c "import verl; print(verl.__version__)"   # 应为 0.7.0
#   2) 小规模跑通:  TRAIN_BATCH_SIZE 调小、total_training_steps 设个位数,确认
#      recipe.one_step_off_policy.main_ppo 能加载 tool_config.yaml + my_custom_hermes
#      + custom_reward_function 不报错(这三处与全异步 agent_loop 的兼容性需实测)。
#   3) 看日志里 rollout/train 是否真并行、staleness 是否稳定在 1 步。
#   4) 对照同步版的收敛曲线；若掉点，先降低staleness或退回同步GRPO。
#   5) 若提速不明显,大概率是外部 API 并发不够 —— 回到 rl/tool_config.yaml 提 rate_limit,
#      而不是继续加 rollout GPU。
# ============================================================================
