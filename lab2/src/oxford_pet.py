import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

class OxfordPetDataset(Dataset):
    def __init__(self, root, mode="train", transform=None, split_dir=None):
        """
        root      : dataset/oxford-iiit-pet/ 的路徑
        mode      : "train", "valid", "test"
        split_dir : Kaggle 競賽資料夾路徑（含 train.txt / val.txt / test_*.txt）
                    若為 None，使用 Oxford 官方 split
        """
        assert mode in ["train", "valid", "test"]
        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_dir = os.path.join(root, "images")
        self.masks_dir  = os.path.join(root, "annotations", "trimaps")

        # 決定 split 清單路徑
        if split_dir is not None:
            if mode == "train":
                list_file = os.path.join(split_dir, "train.txt")
            elif mode == "valid":
                list_file = os.path.join(split_dir, "val.txt")
            else:  # test
                # 自動找 test_*.txt
                candidates = [f for f in os.listdir(split_dir) if f.startswith("test_")]
                assert candidates, f"找不到 test_*.txt in {split_dir}"
                list_file = os.path.join(split_dir, candidates[0])
        else:
            # 使用 Oxford 官方 split
            if mode == "train":
                list_file = os.path.join(root, "annotations", "trainval.txt")
            else:
                list_file = os.path.join(root, "annotations", "test.txt")

        self.filenames = []
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1 and not parts[0].startswith("#"):
                    self.filenames.append(parts[0])   # e.g. "Abyssinian_1"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # 載入圖片
        img_path  = os.path.join(self.images_dir, name + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.mode == "test":
            # test set 沒有 mask，只回傳圖片和原始尺寸
            orig_size = image.size  # (W, H)
            image = self._transform_image(image)
            return image, name, orig_size

        # 載入 trimap mask
        mask_path = os.path.join(self.masks_dir, name + ".png")
        mask = Image.open(mask_path)   # 值為 1, 2, 3

        # 套用資料增強（train）或只做 resize（valid）
        image, mask = self._apply_transforms(image, mask)

        # trimap → binary mask
        # 1 = foreground → 1
        # 2 = background → 0
        # 3 = boundary   → 0  (依規定視為 background)
        mask = np.array(mask, dtype=np.int64)
        binary_mask = (mask == 1).astype(np.float32)   # shape: (H, W)
        binary_mask = torch.tensor(binary_mask).unsqueeze(0)  # (1, H, W)

        return image, binary_mask

    # ------------------------------------------------------------------ #
    #  內部 helper                                                         #
    # ------------------------------------------------------------------ #
    def _apply_transforms(self, image, mask):
        """資料前處理 + Augmentation（只在 train 做增強）"""
        target_size = (256, 256)

        # 1. Resize
        image = TF.resize(image, target_size)
        mask  = TF.resize(mask,  target_size, interpolation=InterpolationMode.NEAREST)

        if self.mode == "train":
            # 2. Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 3. Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # 4. Random Rotation (±30°)
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

            # 5. Color Jitter（只對圖片做，mask 不動）
            color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            )
            image = color_jitter(image)

        # 6. ToTensor + Normalize（ImageNet mean/std）
        image = TF.to_tensor(image)   # [0,1], shape (3, H, W)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        return image, mask

    def _transform_image(self, image):
        """test set 只做 resize + normalize"""
        image = TF.resize(image, (256, 256))
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        return image


# ------------------------------------------------------------------ #
#  DataLoader 工廠函式（給 train.py / evaluate.py / inference.py 呼叫）#
# ------------------------------------------------------------------ #
def get_loader(root, mode, batch_size=8, num_workers=0, shuffle=None, split_dir=None):
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = OxfordPetDataset(root=root, mode=mode, split_dir=split_dir)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


# ------------------------------------------------------------------ #
#  快速測試：直接執行此檔確認能正常跑                                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    root = "dataset/oxford-iiit-pet"  # 從 lab2/ 執行
    
    train_loader = get_loader(root, mode="train", batch_size=4)
    images, masks = next(iter(train_loader))
    print(f"[Train] image shape: {images.shape}")   # (4, 3, 256, 256)
    print(f"[Train] mask  shape: {masks.shape}")    # (4, 1, 256, 256)
    print(f"[Train] mask  unique values: {masks.unique()}")  # tensor([0., 1.])

    valid_loader = get_loader(root, mode="valid", batch_size=4)
    images, masks = next(iter(valid_loader))
    print(f"[Valid] image shape: {images.shape}")
    print(f"[Valid] mask  shape: {masks.shape}")