import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from oxford_pet import get_loader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Binary mask (H, W) → RLE string (column-major / Fortran order).
    foreground=1, background=0
    """
    flat = mask.flatten(order="F")   # column-major
    flat = np.concatenate([[0], flat, [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(runs.astype(str))


SPLIT_DIR_MAP = {
    "unet":          "nycu-2026-spring-dl-lab2-unet",
    "resnet34_unet": "binary-semantic-segmentation-res-net-34-u-net",
}


def run_inference(args):
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

    # ── Test DataLoader ───────────────────────────────────────────────
    test_loader = get_loader(
        args.data_path,
        mode="test",
        batch_size=args.batch_size,
        shuffle=False,
        split_dir=args.split_dir,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    def _forward(imgs):
        """Forward pass + center-crop to 384×384"""
        out = torch.sigmoid(model(imgs))
        oh, ow = out.shape[-2], out.shape[-1]
        th, tw = 384, 384
        ch, cw = (oh - th) // 2, (ow - tw) // 2
        if oh != th or ow != tw:
            out = out[:, :, ch:ch+th, cw:cw+tw]
        return out

    # ── Inference (with TTA: original + hflip + vflip) ───────────────
    rows = []
    with torch.no_grad():
        for images, names, orig_sizes in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs  = _forward(images)
            outputs += _forward(torch.flip(images, [-1])).flip(-1)   # hflip
            outputs += _forward(torch.flip(images, [-2])).flip(-2)   # vflip
            outputs /= 3.0
            preds = (outputs > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for name, pred, (orig_w, orig_h) in zip(names, preds, zip(*orig_sizes)):
                # resize mask 回原始圖片尺寸
                pred_img = Image.fromarray(pred).resize(
                    (orig_w.item(), orig_h.item()), resample=Image.NEAREST
                )
                pred_resized = np.array(pred_img, dtype=np.uint8)
                rle = mask_to_rle(pred_resized)
                rows.append({"image_id": name, "encoded_mask": rle})

    # ── Save CSV ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_path",  type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--split_dir",  type=str, default=None,
                        help="split 資料夾（含 test_*.txt）；預設依 model 自動選擇")
    parser.add_argument("--weight",     type=str, default="saved_models/unet_best.pth")
    parser.add_argument("--output",     type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_inference(args)
