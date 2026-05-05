#!/bin/bash
# Evaluate Task 1 model (CartPole) with seeds 0~19.
python test_model.py \
    --model-path ../LAB5_B11107027_task1.pt \
    --env-name CartPole-v1 \
    --output-dir eval_task1_videos \
    --episodes 20 \
    --seed 0
