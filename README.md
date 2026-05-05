# NYCU Deep Learning Labs

This repository collects my coursework implementations for the NYCU Deep
Learning course.  The projects are written mainly in PyTorch and cover both
computer vision and reinforcement learning workflows, from model
implementation to experiment tracking, evaluation, and final reporting.

## Labs

| Lab | Topic | Main Work | Results / Artifacts |
| --- | --- | --- | --- |
| [Lab 2](lab2) | Binary Semantic Segmentation | Implemented U-Net and ResNet34-UNet from scratch for Oxford-IIIT Pet foreground segmentation. Built dataset loading, preprocessing, BCE + Dice loss training, validation, inference, and Kaggle submission scripts. | U-Net reached best validation Dice `0.9187`. Both U-Net and ResNet34-UNet checkpoints and Kaggle submission pipelines are included. |
| [Lab 5](lab5) | Value-based Reinforcement Learning | Implemented DQN for CartPole and Atari Pong, then extended it with Double DQN, Prioritized Experience Replay, multi-step returns, and Dueling DQN. Used W&B for training curves and evaluation tracking. | CartPole achieved `500.00` average reward over seeds 0-19. Vanilla Pong DQN reached `19.05` average reward. Enhanced DQN reached `19.10` at 600k steps and `20.15` at 1.5M steps. |

## Repository Layout

```text
.
|-- lab2/
|   |-- src/
|   |   |-- models/
|   |   |   |-- unet.py
|   |   |   `-- resnet34_unet.py
|   |   |-- train.py
|   |   |-- evaluate.py
|   |   |-- inference.py
|   |   |-- oxford_pet.py
|   |   `-- utils.py
|   |-- saved_models/
|   |-- requirements.txt
|   `-- Lab2_Binary_Semantic_Segmentation_2026_Spring.pdf
|
`-- lab5/
    |-- LAB5_B11107027_Code/
    |   |-- dqn.py
    |   |-- test_model.py
    |   |-- eval_task1.sh
    |   |-- eval_task2.sh
    |   |-- eval_task3.sh
    |   `-- requirements.txt
    |-- figure/
    |-- report.tex
    |-- LAB5_B11107027.pdf
    `-- LAB5_B11107027.zip
```

## Lab 2: Binary Semantic Segmentation

Lab 2 focuses on foreground segmentation for the Oxford-IIIT Pet dataset.  The
goal is to predict a binary pet mask from an RGB image.

Implemented components:

- U-Net with valid convolutions and skip connections.
- ResNet34-UNet with a ResNet34-style encoder and U-Net decoder.
- Oxford-IIIT Pet dataset loader with mask preprocessing.
- BCE + Dice loss to reduce foreground/background collapse.
- Validation with global Dice and IoU.
- Inference scripts for Kaggle CSV submission generation.

Key result:

- U-Net best validation Dice: `0.9187`.
- Generated 739-image Kaggle prediction files for both U-Net and
  ResNet34-UNet.

## Lab 5: Value-based Reinforcement Learning

Lab 5 studies DQN and enhanced DQN methods on CartPole and Atari Pong.

Implemented components:

- Vanilla DQN with replay buffer, target network, Bellman target computation,
  and epsilon-greedy exploration.
- Atari preprocessing with grayscale conversion, 84 x 84 resizing, and
  4-frame stacking.
- CNN Q-network for Pong.
- Double DQN for decoupled action selection and target evaluation.
- Prioritized Experience Replay with importance-sampling weights.
- Multi-step return support.
- Dueling DQN architecture.
- W&B logging for reward curves, evaluation scores, and ablation comparisons.

Key results:

| Task | Environment | Result |
| --- | --- | --- |
| Task 1 | CartPole-v1 | Average reward `500.00`, min `500.0`, max `500.0` over seeds 0-19. |
| Task 2 | ALE/Pong-v5 | Vanilla DQN average reward `19.05` over seeds 26-45. |
| Task 3 | ALE/Pong-v5 | Enhanced DQN average reward `19.10` at 600k steps and `20.15` at 1.5M steps. |

The Lab 5 report also includes required ablations for no PER, no Double DQN,
and no multi-step return, plus additional bonus analysis for dueling networks,
learning-rate decay, n-step choices, soft target updates, and target update
frequency.

## Reproducing Experiments

Each lab has its own dependency file.

Lab 2:

```bash
cd lab2
pip install -r requirements.txt
python src/train.py --model unet
python src/evaluate.py --model unet --weight saved_models/unet_best.pth
```

Lab 5:

```bash
cd lab5/LAB5_B11107027_Code
pip install -r requirements.txt
bash eval_task1.sh
bash eval_task2.sh
bash eval_task3.sh
```

## Notes

- The repository is organized by lab folder so each assignment can be inspected
  independently.
- Large datasets, checkpoints, and generated submission artifacts may be kept
  locally or in submission packages depending on course requirements.
- The code is coursework-oriented and prioritizes reproducibility of the lab
  results and reports.
