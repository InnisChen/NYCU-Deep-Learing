from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


def save_tensor_image(image: torch.Tensor, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    save_image(denormalize(image), str(path))


def save_tensor_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    ensure_dir(Path(path).parent)
    save_image(make_grid(denormalize(images), nrow=nrow), str(path))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def copy_to_backup(path: str | Path, backup_dir: Optional[str | Path]) -> None:
    if not backup_dir:
        return
    path = Path(path)
    backup_dir = ensure_dir(backup_dir)
    if path.exists():
        shutil.copy2(path, backup_dir / path.name)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    ema: Optional[Any],
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    best_loss: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(Path(path).parent)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "config": config,
        "best_loss": best_loss,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def backup_directory(src_dir: str | Path, dst_dir: Optional[str | Path]) -> None:
    if not dst_dir:
        return
    src_dir = Path(src_dir)
    dst_dir = ensure_dir(dst_dir)
    for path in src_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)
