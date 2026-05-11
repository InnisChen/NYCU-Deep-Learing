# Lab 6: Conditional DDPM for i-CLEVR

This lab implements a conditional Denoising Diffusion Probabilistic Model
(DDPM) for generating i-CLEVR images from multi-label object conditions.  The
model is written in PyTorch and trained from scratch without using diffusion
frameworks such as `diffusers`.

## Task

Given a set of object labels such as `["red sphere", "cyan cylinder",
"cyan cube"]`, the model generates a synthetic 64 x 64 RGB image containing the
specified objects.  The generated images are evaluated by the provided
pretrained ResNet18 evaluator on both `test.json` and `new_test.json`.

## Final Results

| Split | Accuracy |
| --- | ---: |
| `test.json` | `0.875000` |
| `new_test.json` | `0.857143` |

Both scores are above the full-score threshold of `0.800`.

## Method

The implementation uses a conditional DDPM with the following choices:

- A U-Net noise predictor with residual blocks and self-attention.
- Sinusoidal timestep embeddings.
- Multi-hot condition vectors projected by an MLP and injected into residual
  blocks together with timestep embeddings.
- Cosine beta schedule over 1000 diffusion steps.
- Huber loss for noise prediction.
- Classifier-free guidance during sampling by randomly dropping conditions
  during training.
- Exponential moving average (EMA) weights for sampling and validation.
- DDIM sampling for faster generation.
- Periodic evaluator-based validation for checkpoint selection.

The pretrained evaluator is only used as a frozen external metric.  Its source
code and checkpoint are not modified.

## Repository Layout

```text
lab6/
|-- src/
|   |-- dataset.py       # i-CLEVR dataset and condition utilities
|   |-- diffusion.py     # DDPM / DDIM diffusion process
|   |-- ema.py           # EMA parameter tracking
|   |-- evaluate.py      # Evaluation with the provided evaluator
|   |-- models.py        # Conditional U-Net
|   |-- sample.py        # Image generation script
|   |-- train.py         # Training script
|   |-- utils.py         # Common utilities
|   `-- __init__.py
|-- images/
|   |-- test/            # 32 generated images for test.json
|   |-- new_test/        # 32 generated images for new_test.json
|   |-- test_grid.png
|   |-- new_test_grid.png
|   `-- denoising_process.png
|-- file/file/           # Provided metadata and evaluator files
|-- requirements.txt
`-- DL_LAB6_B11107027_陳映宬_report.pdf
```

Large training datasets and generated training checkpoints are intentionally
excluded from the repository.

## Setup

Install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and torchvision should be installed separately according to the target
CUDA environment.  On Colab, the preinstalled PyTorch environment is sufficient.

The i-CLEVR training images should be placed outside Git tracking, for example:

```text
/content/data/iclevr/
```

The provided lab files should be available under:

```text
file/file/
|-- train.json
|-- test.json
|-- new_test.json
|-- objects.json
|-- evaluator.py
`-- checkpoint.pth
```

## Training

Example Colab training command:

```bash
python -m src.train \
  --data-root /content/data/iclevr \
  --meta-dir file/file \
  --save-dir /content/lab6_runs/baseline \
  --backup-dir /content/drive/MyDrive/lab6/checkpoints \
  --epochs 300 \
  --batch-size 128 \
  --lr 2e-4
```

Checkpoints are saved every 50 epochs by default.  Cloud backup can be disabled
by omitting `--backup-dir` or setting `--backup-every 0`.

## Sampling

Generate images for both testing splits:

```bash
python -m src.sample \
  --meta-dir file/file \
  --ckpt /content/drive/MyDrive/lab6/checkpoints/best_ema.pt \
  --out-dir /content/images \
  --split both \
  --sample-steps 100 \
  --cfg-scale 2.0
```

The generated image names follow the order of `test.json` and `new_test.json`:

```text
images/test/0.png ... images/test/31.png
images/new_test/0.png ... images/new_test/31.png
```

The script also saves `test_grid.png`, `new_test_grid.png`, and a denoising
process grid for the required label set.

## Evaluation

Evaluate the generated images with the provided evaluator:

```bash
python -m src.evaluate \
  --meta-dir file/file \
  --image-dir /content/images \
  --split both
```

The final evaluation output used in the report is:

```text
test.json     : 0.875000
new_test.json : 0.857143
```

## Notes

- The dataset is not included in this repository.
- Training checkpoints and runtime outputs should not be committed.
- `lab6_colab.ipynb` is only a personal Colab helper and is not required for
  submission.
