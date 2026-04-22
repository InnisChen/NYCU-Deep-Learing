#!/bin/bash
# Evaluate Task 3 model snapshots (Pong) with seeds 0~19
# Run the snapshot that corresponds to your best result for grading

STUDENT_ID="B11107027"

for STEPS in 600000 1000000 1500000 2000000 2500000; do
    MODEL="LAB5_${STUDENT_ID}_task3_${STEPS}.pt"
    echo "=== Evaluating: $MODEL ==="
    python test_model.py \
        --model-path "$MODEL" \
        --output-dir "eval_task3_videos/${STEPS}" \
        --episodes 20 \
        --seed 0
done

echo "=== Evaluating best model ==="
python test_model.py \
    --model-path "LAB5_${STUDENT_ID}_task3_best.pt" \
    --output-dir "eval_task3_videos/best" \
    --episodes 20 \
    --seed 0
