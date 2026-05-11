from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from .dataset import ConditionDataset
from .evaluate import build_summary, find_ordered_image, format_results_table


def build_resnet18_evaluator(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Linear(512, 24), nn.Sigmoid())
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def compute_acc(out: torch.Tensor, onehot_labels: torch.Tensor) -> float:
    batch_size = out.size(0)
    acc = 0
    total = 0
    for i in range(batch_size):
        k = int(onehot_labels[i].sum().item())
        total += k
        _, outi = out[i].topk(k)
        _, li = onehot_labels[i].topk(k)
        for j in outi:
            if j in li:
                acc += 1
    return acc / total


def load_split_images(image_dir: str | Path, split: str, count: int) -> torch.Tensor:
    transform = image_transform()
    split_dir = Path(image_dir) / split
    images = []
    for idx in range(count):
        path = find_ordered_image(split_dir, idx)
        images.append(transform(Image.open(path).convert("RGB")))
    return torch.stack(images, dim=0)


@torch.no_grad()
def evaluate_split(args, model: nn.Module, split: str, device: torch.device) -> Tuple[str, float]:
    dataset = ConditionDataset(args.meta_dir, split=split)
    labels = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    images = load_split_images(args.image_dir, split, len(dataset))
    logits = model(images.to(device))
    acc = compute_acc(logits.cpu(), labels.cpu())
    return split, float(acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lab6 images locally without modifying the provided evaluator.")
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--split", type=str, default="both", choices=["test", "new_test", "both"])
    parser.add_argument("--checkpoint", type=str, default="file/file/checkpoint.pth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--score-threshold", type=float, default=0.8)
    args = parser.parse_args()
    args.rerank_candidates = False

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    model = build_resnet18_evaluator(args.checkpoint, device)
    splits = ["test", "new_test"] if args.split == "both" else [args.split]
    results: List[Tuple[str, float]] = [evaluate_split(args, model, split, device) for split in splits]

    summary = build_summary(results, args.score_threshold)
    print(format_results_table(args, results, summary))
    print(f"\nLocal evaluator device: {device}")
    print("Note: this script mirrors the provided evaluator on CPU/GPU without editing file/file/evaluator.py.")


if __name__ == "__main__":
    main()
