#!/bin/bash
# Evaluate Task 2 model (Pong) with seeds 26~45.
python test_model.py \
    --model-path ../LAB5_B11107027_task2.pt \
    --env-name ALE/Pong-v5 \
    --output-dir eval_task2_videos \
    --episodes 20 \
    --seed 26
