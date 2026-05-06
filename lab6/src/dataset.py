from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_object_map(meta_dir: str | Path) -> Dict[str, int]:
    return load_json(Path(meta_dir) / "objects.json")


def labels_to_multihot(labels: Sequence[str], object_map: Dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(object_map), dtype=torch.float32)
    for label in labels:
        if label not in object_map:
            raise KeyError(f"Unknown i-CLEVR label: {label}")
        vector[object_map[label]] = 1.0
    return vector


def default_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_image_index(data_root: str | Path, image_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    root = Path(image_dir) if image_dir is not None else Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    image_paths = list(root.rglob("*.png")) + list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return {path.name: path for path in image_paths}


class IClevrDataset(Dataset):
    """Training dataset for conditional DDPM on i-CLEVR."""

    def __init__(
        self,
        data_root: str | Path,
        meta_dir: str | Path,
        image_size: int = 64,
        image_dir: Optional[str | Path] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.meta_dir = Path(meta_dir)
        self.object_map = load_object_map(self.meta_dir)
        train_json = load_json(self.meta_dir / "train.json")
        self.items: List[Tuple[str, List[str]]] = [(name, list(labels)) for name, labels in train_json.items()]
        self.image_index = build_image_index(self.data_root, image_dir=image_dir)
        self.transform = transform or default_image_transform(image_size)

        missing = [name for name, _ in self.items if name not in self.image_index]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} training images. First missing: {preview}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        filename, labels = self.items[index]
        image = Image.open(self.image_index[filename]).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = labels_to_multihot(labels, self.object_map)
        return image_tensor, label_tensor


class ConditionDataset(Dataset):
    """Condition-only dataset for test.json and new_test.json."""

    def __init__(self, meta_dir: str | Path, split: str = "test") -> None:
        self.meta_dir = Path(meta_dir)
        self.split = split
        self.object_map = load_object_map(self.meta_dir)
        if split not in {"test", "new_test"}:
            raise ValueError("split must be 'test' or 'new_test'")
        self.conditions: List[List[str]] = [list(labels) for labels in load_json(self.meta_dir / f"{split}.json")]

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int):
        labels = self.conditions[index]
        return labels_to_multihot(labels, self.object_map), labels, index


def labels_batch_to_multihot(labels_batch: Iterable[Sequence[str]], object_map: Dict[str, int]) -> torch.Tensor:
    return torch.stack([labels_to_multihot(labels, object_map) for labels in labels_batch], dim=0)

