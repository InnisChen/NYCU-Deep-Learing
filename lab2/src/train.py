import os
import argparse
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from oxford_pet import get_loader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_score, bce_dice_loss


SPLIT_DIR_MAP = {
    "unet":          "nycu-2026-spring-dl-lab2-unet",
    "resnet34_unet": "binary-semantic-segmentation-res-net-34-u-net",
}


def train(args):
    if args.split_dir is None:
        args.split_dir = SPLIT_DIR_MAP[args.model]
        print(f"Auto-selected split_dir: {args.split_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── DataLoader ───────────────────────────────────────────────────
    train_loader = get_loader(args.data_path, mode="train", batch_size=args.batch_size, split_dir=args.split_dir)
    valid_loader = get_loader(args.data_path, mode="valid", batch_size=args.batch_size, split_dir=args.split_dir)
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # ── Model ────────────────────────────────────────────────────────
    if args.model == "unet":
        model = UNet()
    else:
        model = ResNet34UNet()
    model = model.to(device)
    print(f"Model: {args.model}  |  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss & Optimizer ─────────────────────────────────────────────
    criterion = bce_dice_loss   # BCE + Dice，避免模型 collapse 到全背景
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training loop ────────────────────────────────────────────────
    best_dice = 0.0
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, f"{args.model}_best.pth")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks  = masks.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                val_dice += dice_score(outputs, masks)

        val_dice /= len(valid_loader)
        scheduler.step(val_dice)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_dice={val_dice:.4f}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_file)
            print(f"  → Best model saved (dice={best_dice:.4f})")

    print(f"\nTraining done. Best Val Dice: {best_dice:.4f}")
    print(f"Model saved to: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,   default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_path",     type=str,   default="dataset/oxford-iiit-pet")
    parser.add_argument("--split_dir",     type=str,   default=None,
                        help="Kaggle 競賽 split 資料夾（含 train.txt/val.txt/test_*.txt）")
    parser.add_argument("--save_path",     type=str,   default="saved_models")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
