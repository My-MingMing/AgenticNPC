#!/bin/bash
# ============================================================================
# NPC对话模型 - 三轮数据飞轮自动编排
#
# 每轮迭代：
#   1. 从上一轮失败case扩展行为树 → 生成更难的对抗场景
#   2. Agent在模拟环境中求解 → Rubric评估 → 过滤训练数据
#   3. RL训练 1 epoch（固定超参，靠数据变难驱动能力提升）
#   4. Merge checkpoint → 作为下一轮的模型初始化
#
# 用法:
#   bash run_flywheel.sh              # 默认 8xH200，完整3轮
#   bash run_flywheel.sh a100         # 4xA100
#   bash run_flywheel.sh h200 2       # 从Round 2恢复（断点续训）
# ============================================================================

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
HW_PROFILE="${1:-h200}"
START_ROUND="${2:-1}"
TOTAL_ROUNDS=3

# 初始模型（Phase 1 Warm-up的输出）
WARMUP_CKPT="${PROJECT_ROOT}/checkpoints/warmup/merge"
if [ ! -d "${WARMUP_CKPT}" ]; then
    echo "Error: Warm-up checkpoint not found at ${WARMUP_CKPT}"
    echo "Please run Phase 1 (run_warmup_training.sh) first."
    exit 1
fi

echo "============================================================"
echo "NPC Dialogue Data Flywheel - ${TOTAL_ROUNDS} Rounds"
echo "Hardware: ${HW_PROFILE}, Starting from Round ${START_ROUND}"
echo "Warm-up checkpoint: ${WARMUP_CKPT}"
echo "============================================================"

for ROUND in $(seq ${START_ROUND} ${TOTAL_ROUNDS}); do
    echo ""
    echo "============================================================"
    echo "  ROUND ${ROUND} / ${TOTAL_ROUNDS}"
    echo "============================================================"

    # 确定本轮模型初始化路径
    if [ ${ROUND} -eq 1 ]; then
        ROUND_MODEL="${WARMUP_CKPT}"
    else
        PREV_ROUND=$((ROUND - 1))
        ROUND_MODEL="${PROJECT_ROOT}/checkpoints/round${PREV_ROUND}/merge"
        if [ ! -d "${ROUND_MODEL}" ]; then
            echo "Error: Round ${PREV_ROUND} checkpoint not found at ${ROUND_MODEL}"
            exit 1
        fi
    fi

    ROUND_OUTPUT="${PROJECT_ROOT}/checkpoints/round${ROUND}"
    ROUND_DATA="${PROJECT_ROOT}/data/round${ROUND}"
    mkdir -p "${ROUND_DATA}" "${ROUND_OUTPUT}"

    echo "  Model: ${ROUND_MODEL}"
    echo "  Data:  ${ROUND_DATA}"
    echo "  Output: ${ROUND_OUTPUT}"

    # ─────────────────────────────────────────────────────────────────
    # Step 1: 数据合成（行为树扩展 + 飞轮增难）
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "  [Step 1/${ROUND}] Generating training data (flywheel expansion)..."

    if [ ${ROUND} -eq 1 ]; then
        # Round 1: 使用原始种子直接生成
        cd "${PROJECT_ROOT}/scripts"
        python run_npc_data_gen.py \
            --output "${ROUND_DATA}/generated_tasks.jsonl"
    else
        # Round 2+: 从上一轮失败case扩展行为树
        cd "${PROJECT_ROOT}/scripts"
        python expand_from_failures.py \
            --prev_eval "${PROJECT_ROOT}/data/round${PREV_ROUND}/evaluated_tasks.jsonl" \
            --personas "${PROJECT_ROOT}/configs/npc_personas.jsonl" \
            --output "${ROUND_DATA}/expanded_personas.jsonl" \
            --round ${ROUND}

        python run_npc_data_gen.py \
            --personas "${ROUND_DATA}/expanded_personas.jsonl" \
            --output "${ROUND_DATA}/generated_tasks.jsonl"
    fi

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Agent求解
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "  [Step 2/${ROUND}] Solving tasks with current model..."
    cd "${PROJECT_ROOT}/scripts"
    python run_npc_solve_task.py \
        --input "${ROUND_DATA}/generated_tasks.jsonl" \
        --output "${ROUND_DATA}/solved_tasks.jsonl" \
        --repeats 2 --max_turns 10

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Rubric评估
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "  [Step 3/${ROUND}] Evaluating solutions..."
    cd "${PROJECT_ROOT}/scripts"
    python run_npc_rubrics.py \
        --input "${ROUND_DATA}/solved_tasks.jsonl" \
        --output "${ROUND_DATA}/evaluated_tasks.jsonl" \
        --threshold 0.6

    # ─────────────────────────────────────────────────────────────────
    # Step 4: 过滤+格式化训练数据
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "  [Step 4/${ROUND}] Preparing training data..."
    cd "${PROJECT_ROOT}/scripts"
    python make_filtered_npc_data.py \
        --input "${ROUND_DATA}/evaluated_tasks.jsonl" \
        --output_train "${ROUND_DATA}/train.json" \
        --output_val "${ROUND_DATA}/val.json"

    python "${PROJECT_ROOT}/rl/convert_parquet.py" \
        --input "${ROUND_DATA}/train.json" \
        --output "${ROUND_DATA}/train.parquet"
    python "${PROJECT_ROOT}/rl/convert_parquet.py" \
        --input "${ROUND_DATA}/val.json" \
        --output "${ROUND_DATA}/val.parquet"

    # ─────────────────────────────────────────────────────────────────
    # Step 5: RL训练 (1 epoch, 固定超参)
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "  [Step 5/${ROUND}] RL Training (1 epoch, fixed hyperparams)..."
    cd "${PROJECT_ROOT}/scripts"
    MODEL_PATH="${ROUND_MODEL}" \
    DATA_PATH="${ROUND_DATA}/train.parquet" \
    VAL_DATA_PATH="${ROUND_DATA}/val.parquet" \
    OUTPUT_DIR="${ROUND_OUTPUT}" \
    bash run_npc_training.sh ${HW_PROFILE}

    # Merge后的模型自动在 ${ROUND_OUTPUT}/merge
    echo ""
    echo "  [Round ${ROUND} COMPLETE]"
    echo "  Merged model: ${ROUND_OUTPUT}/merge"
    echo ""
done

echo "============================================================"
echo "ALL ${TOTAL_ROUNDS} ROUNDS COMPLETE!"
echo ""
echo "Final model: ${PROJECT_ROOT}/checkpoints/round${TOTAL_ROUNDS}/merge"
echo ""
echo "To deploy:"
echo "  python -m sglang.launch_server \\"
echo "    --model-path ${PROJECT_ROOT}/checkpoints/round${TOTAL_ROUNDS}/merge \\"
echo "    --port 8000 --tp 1"
echo "============================================================"
