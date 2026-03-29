import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from oxford_pet import get_loader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


SPLIT_DIR_MAP = {
    "unet":          "nycu-2026-spring-dl-lab2-unet",
    "resnet34_unet": "binary-semantic-segmentation-res-net-34-u-net",
}


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def run_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.split_dir is None:
        args.split_dir = SPLIT_DIR_MAP[args.model]
    print(f"Split dir: {args.split_dir}")

    # ── Model ────────────────────────────────────────────────────────
    if args.model == "unet":
        model = UNet()
    else:
        model = ResNet34UNet()

    assert os.path.exists(args.weight), f"Weight not found: {args.weight}"
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded weights: {args.weight}")

    # ── DataLoader ───────────────────────────────────────────────────
    loader = get_loader(
        args.data_path,
        mode="valid",
        batch_size=args.batch_size,
        shuffle=False,
        split_dir=args.split_dir,
    )
    print(f"Val samples: {len(loader.dataset)}")

    # ── Evaluation ───────────────────────────────────────────────────
    total_intersection = 0
    total_pred = 0
    total_gt   = 0
    iou_list, n_samples = [], 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))          # (B,1,H,W)
            oh, ow = outputs.shape[-2], outputs.shape[-1]
            mh, mw = masks.shape[-2], masks.shape[-1]
            ch, cw = (oh - mh) // 2, (ow - mw) // 2
            if oh != mh or ow != mw:
                outputs = outputs[:, :, ch:ch+mh, cw:cw+mw]
            preds = (outputs > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            targets = masks.squeeze(1).numpy().astype(np.uint8)

            for pred, target in zip(preds, targets):
                total_intersection += (pred * target).sum()
                total_pred += pred.sum()
                total_gt   += target.sum()
                iou_list.append(iou_score(pred, target))
                n_samples += 1

    global_dice = (2.0 * total_intersection) / (total_pred + total_gt + 1e-8)
    print(f"Global Dice (Kaggle): {global_dice:.4f}  |  IoU: {np.mean(iou_list):.4f}")
    print(f"Samples: {n_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_path",  type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--split_dir",  type=str, default=None,
                        help="split 資料夾；預設依 model 自動選擇")
    parser.add_argument("--weight",     type=str, default="saved_models/unet_best.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_evaluate(args)
