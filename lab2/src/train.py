import os
import argparse
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from oxford_pet import get_loader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_components, bce_dice_loss


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
    train_loader = get_loader(args.data_path, mode="train", batch_size=args.batch_size, split_dir=args.split_dir, num_workers=args.num_workers)
    valid_loader = get_loader(args.data_path, mode="valid", batch_size=args.batch_size, split_dir=args.split_dir, num_workers=args.num_workers)
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150, 175], gamma=0.5
    )

    # ── Training loop ────────────────────────────────────────────────
    best_dice = 0.0
    start_epoch = 1
    os.makedirs(args.save_path, exist_ok=True)
    save_file   = os.path.join(args.save_path, f"{args.model}_best.pth")
    ckpt_file   = os.path.join(args.save_path, f"{args.model}_checkpoint.pth")

    # Resume
    if args.resume and os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_dice   = ckpt["best_dice"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  best_dice={best_dice:.4f}")
    elif args.resume:
        print(f"Checkpoint not found at {ckpt_file}, starting from scratch.")

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Training"):
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                # center-crop to match mask size (reflection pad 補的邊)
                oh, ow = outputs.shape[-2], outputs.shape[-1]
                mh, mw = masks.shape[-2], masks.shape[-1]
                ch, cw = (oh - mh) // 2, (ow - mw) // 2
                outputs = outputs[:, :, ch:ch+mh, cw:cw+mw]
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
        total_intersection = 0.0
        total_pred = 0.0
        total_gt   = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks  = masks.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                oh, ow = outputs.shape[-2], outputs.shape[-1]
                mh, mw = masks.shape[-2], masks.shape[-1]
                ch, cw = (oh - mh) // 2, (ow - mw) // 2
                outputs = outputs[:, :, ch:ch+mh, cw:cw+mw]
                i, p, g = dice_components(outputs, masks)
                total_intersection += i
                total_pred += p
                total_gt   += g

        val_dice = (2.0 * total_intersection) / (total_pred + total_gt + 1e-8)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_dice={val_dice:.4f}")

        # Save best model locally (fast, no Drive I/O)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_file)
            print(f"  → Best model saved (dice={best_dice:.4f})")

        # Save periodic snapshot (every 40 epochs) + backup to Drive
        if epoch % 40 == 0:
            ckpt_epoch_file = os.path.join(args.save_path, f"{args.model}_epoch{epoch}.pth")
            ckpt_data = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
            }
            torch.save(ckpt_data, ckpt_epoch_file)
            torch.save(ckpt_data, ckpt_file)
            print(f"  → Snapshot saved (epoch={epoch})")

            # Backup checkpoint + best model to Drive
            if args.backup_path:
                import shutil
                os.makedirs(args.backup_path, exist_ok=True)
                shutil.copy(ckpt_epoch_file, os.path.join(args.backup_path, f"{args.model}_epoch{epoch}.pth"))
                shutil.copy(ckpt_file,       os.path.join(args.backup_path, f"{args.model}_checkpoint.pth"))
                if os.path.exists(save_file):
                    shutil.copy(save_file,   os.path.join(args.backup_path, f"{args.model}_best.pth"))
                print(f"  → Backed up to Drive (epoch={epoch})")

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
    parser.add_argument("--backup_path",   type=str,   default=None,
                        help="每 40 epoch 備份 checkpoint + best model 到此路徑（Drive）")
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--resume",        action="store_true",
                        help="從上次的 checkpoint 繼續訓練")
    args = parser.parse_args()
    train(args)
