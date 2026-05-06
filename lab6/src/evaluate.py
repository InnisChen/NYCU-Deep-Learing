from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from .dataset import ConditionDataset
from .utils import denormalize, ensure_dir


def load_evaluator(meta_dir: str | Path):
    if not torch.cuda.is_available():
        raise RuntimeError("The provided evaluator calls .cuda(); run evaluation on a CUDA runtime.")
    meta_dir = Path(meta_dir).resolve()
    evaluator_path = meta_dir / "evaluator.py"
    old_cwd = Path.cwd()
    try:
        os.chdir(meta_dir)
        spec = importlib.util.spec_from_file_location("lab6_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import evaluator from {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluation_model()
    finally:
        os.chdir(old_cwd)


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_split_images(image_dir: str | Path, split: str, count: int) -> torch.Tensor:
    transform = image_transform()
    split_dir = Path(image_dir) / split
    images = []
    for idx in range(count):
        path = split_dir / f"{idx:03d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing generated image: {path}")
        images.append(transform(Image.open(path).convert("RGB")))
    return torch.stack(images, dim=0)


def per_image_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = []
    for output, target in zip(logits.cpu(), labels.cpu()):
        k = int(target.sum().item())
        pred_idx = output.topk(k).indices
        gt_idx = target.topk(k).indices
        scores.append(sum(1 for idx in pred_idx if idx in gt_idx) / max(k, 1))
    return torch.tensor(scores)


def rerank_candidates(evaluator, image_dir: str | Path, split: str, labels: torch.Tensor, num_candidates: int) -> None:
    transform = image_transform()
    image_dir = Path(image_dir)
    candidate_root = image_dir / f"{split}_candidates"
    selected_dir = ensure_dir(image_dir / split)
    selected_images = []

    for idx in range(labels.shape[0]):
        cand_dir = candidate_root / f"{idx:03d}"
        candidates = []
        for cand_idx in range(num_candidates):
            path = cand_dir / f"{cand_idx:03d}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing candidate image: {path}")
            candidates.append(transform(Image.open(path).convert("RGB")))
        batch = torch.stack(candidates, dim=0).cuda()
        batch_labels = labels[idx : idx + 1].repeat(num_candidates, 1).cuda()
        with torch.no_grad():
            logits = evaluator.resnet18(batch)
        scores = per_image_accuracy(logits, batch_labels)
        best_idx = int(scores.argmax().item())
        best_src = cand_dir / f"{best_idx:03d}.png"
        best_dst = selected_dir / f"{idx:03d}.png"
        shutil.copy2(best_src, best_dst)
        selected_images.append(transform(Image.open(best_dst).convert("RGB")))

    grid = torch.stack(selected_images, dim=0)
    save_image(denormalize(grid), image_dir / f"{split}_grid_reranked.png", nrow=8)


def evaluate_split(args, evaluator, split: str) -> Tuple[str, float]:
    dataset = ConditionDataset(args.meta_dir, split=split)
    labels = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    if args.rerank_candidates:
        rerank_candidates(evaluator, args.image_dir, split, labels, args.num_candidates)
    images = load_split_images(args.image_dir, split, len(dataset)).cuda()
    labels = labels.cuda()
    acc = evaluator.eval(images, labels)
    return split, float(acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated i-CLEVR images with the provided classifier.")
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--image-dir", type=str, default="/content/lab6_outputs")
    parser.add_argument("--split", type=str, default="both", choices=["test", "new_test", "both"])
    parser.add_argument("--rerank-candidates", action="store_true")
    parser.add_argument("--num-candidates", type=int, default=4)
    args = parser.parse_args()

    evaluator = load_evaluator(args.meta_dir)
    splits = ["test", "new_test"] if args.split == "both" else [args.split]
    results: List[Tuple[str, float]] = []
    for split in splits:
        results.append(evaluate_split(args, evaluator, split))
    for split, acc in results:
        print(f"{split}_acc={acc:.6f}")


if __name__ == "__main__":
    main()

