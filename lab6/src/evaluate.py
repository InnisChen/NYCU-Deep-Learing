from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from .dataset import ConditionDataset
from .utils import denormalize, ensure_dir


def output_image_name(index: int) -> str:
    return f"{index}.png"


def find_ordered_image(directory: str | Path, index: int) -> Path:
    directory = Path(directory)
    candidates = [directory / output_image_name(index), directory / f"{index:03d}.png"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing generated image for index {index}: expected {candidates[0]} or {candidates[1]}")


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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*torch.load.*", category=FutureWarning)
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
        path = find_ordered_image(split_dir, idx)
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
        cand_dir = candidate_root / str(idx)
        if not cand_dir.exists():
            cand_dir = candidate_root / f"{idx:03d}"
        candidates = []
        for cand_idx in range(num_candidates):
            path = find_ordered_image(cand_dir, cand_idx)
            candidates.append(transform(Image.open(path).convert("RGB")))
        batch = torch.stack(candidates, dim=0).cuda()
        batch_labels = labels[idx : idx + 1].repeat(num_candidates, 1).cuda()
        with torch.no_grad():
            logits = evaluator.resnet18(batch)
        scores = per_image_accuracy(logits, batch_labels)
        best_idx = int(scores.argmax().item())
        best_src = find_ordered_image(cand_dir, best_idx)
        best_dst = selected_dir / output_image_name(idx)
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


def display_name(split: str) -> str:
    return "new_test.json" if split == "new_test" else "test.json"


def build_summary(results: List[Tuple[str, float]], threshold: float) -> Dict[str, object]:
    scores = {split: acc for split, acc in results}
    passes = {split: acc >= threshold for split, acc in results}
    average = sum(scores.values()) / max(1, len(scores))
    return {
        "scores": scores,
        "passes": passes,
        "average": average,
        "threshold": threshold,
    }


def format_results_table(args, results: List[Tuple[str, float]], summary: Dict[str, object]) -> str:
    threshold = float(summary["threshold"])
    average = float(summary["average"])
    rows = [(display_name(split), f"{acc:.6f}", f"PASS >= {threshold:.3f}" if acc >= threshold else f"FAIL < {threshold:.3f}") for split, acc in results]
    if len(results) > 1:
        rows.append(("Average", f"{average:.6f}", "checkpoint selection"))

    headers = ("Split", "Accuracy", "Status")
    widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(3)]

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row(values) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        "=" * 72,
        " Lab6 Conditional DDPM Evaluation",
        "=" * 72,
        f" Image directory : {args.image_dir}",
        f" Meta directory  : {args.meta_dir}",
        f" Rerank          : {'enabled' if args.rerank_candidates else 'disabled'}",
        f" Full-score bar  : accuracy >= {threshold:.3f}",
        "",
        border(),
        row(headers),
        border(),
    ]
    lines.extend(row(values) for values in rows)
    lines.append(border())
    return "\n".join(lines)


def save_results_json(path: str | Path, args, results: List[Tuple[str, float]], summary: Dict[str, object]) -> None:
    path = Path(path)
    payload = {
        "image_dir": str(args.image_dir),
        "meta_dir": str(args.meta_dir),
        "rerank_candidates": bool(args.rerank_candidates),
        "num_candidates": int(args.num_candidates),
        "threshold": float(summary["threshold"]),
        "average": float(summary["average"]),
        "results": [
            {
                "split": split,
                "file": display_name(split),
                "accuracy": acc,
                "passed_full_score_bar": bool(acc >= float(summary["threshold"])),
            }
            for split, acc in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated i-CLEVR images with the provided classifier.")
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--image-dir", type=str, default="/content/images")
    parser.add_argument("--split", type=str, default="both", choices=["test", "new_test", "both"])
    parser.add_argument("--rerank-candidates", action="store_true")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--no-save-results", action="store_true")
    args = parser.parse_args()

    evaluator = load_evaluator(args.meta_dir)
    splits = ["test", "new_test"] if args.split == "both" else [args.split]
    results: List[Tuple[str, float]] = []
    for split in splits:
        results.append(evaluate_split(args, evaluator, split))

    summary = build_summary(results, args.score_threshold)
    report_text = format_results_table(args, results, summary)
    print(report_text)

    if not args.no_save_results:
        output_dir = ensure_dir(args.image_dir)
        txt_path = output_dir / "evaluation_results.txt"
        json_path = output_dir / "evaluation_results.json"
        txt_path.write_text(report_text + "\n", encoding="utf-8")
        save_results_json(json_path, args, results, summary)
        saved_paths = [txt_path, json_path]
        print("\nSaved evaluation summary:")
        for path in saved_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()
