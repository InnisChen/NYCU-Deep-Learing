from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import IClevrDataset, labels_batch_to_multihot, load_object_map
from .diffusion import GaussianDiffusion
from .ema import EMA
from .models import ConditionalUNet
from .utils import (
    copy_to_backup,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_tensor_grid,
    set_seed,
)


def parse_int_tuple(value: str):
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def maybe_init_wandb(args):
    if not args.wandb_run_name:
        return None
    try:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), save_code=True)
        return wandb
    except Exception as exc:  # pragma: no cover - logging fallback
        print(f"W&B disabled: {exc}")
        return None


def build_model(args) -> ConditionalUNet:
    return ConditionalUNet(
        image_size=args.image_size,
        num_classes=24,
        base_channels=args.base_channels,
        channel_mults=parse_int_tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=parse_int_tuple(args.attention_resolutions),
        dropout=args.dropout,
    )


@torch.no_grad()
def save_preview(model, diffusion, ema, object_map, args, device, epoch: int) -> None:
    preview_conditions = [
        ["gray cube"],
        ["red cube"],
        ["blue sphere"],
        ["yellow cylinder"],
        ["red sphere", "cyan cylinder", "cyan cube"],
        ["purple cube", "green sphere"],
        ["brown cylinder", "yellow cube"],
        ["cyan cube", "red sphere"],
    ]
    labels = labels_batch_to_multihot(preview_conditions, object_map).to(device)
    ema.store(model)
    ema.copy_to(model)
    model.eval()
    images = diffusion.ddim_sample(
        model,
        labels,
        image_size=args.image_size,
        sample_steps=min(args.preview_sample_steps, args.timesteps),
        cfg_scale=args.preview_cfg_scale,
        eta=0.0,
    )
    ema.restore(model)
    save_tensor_grid(images, Path(args.save_dir) / "previews" / f"epoch_{epoch:03d}.png", nrow=4)


def save_ema_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss):
    ema.store(model)
    ema.copy_to(model)
    save_checkpoint(path, model, optimizer, scheduler, scaler, None, epoch, global_step, config, best_loss)
    ema.restore(model)


def train(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset = IClevrDataset(
        data_root=args.data_root,
        meta_dir=args.meta_dir,
        image_size=args.image_size,
        image_dir=args.image_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"Training images: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    model = build_model(args).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=args.min_lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, decay=args.ema_decay)

    save_dir = ensure_dir(args.save_dir)
    object_map = load_object_map(args.meta_dir)
    print(f"Model parameters: {count_parameters(model):,}")

    start_epoch = 1
    global_step = 0
    best_loss = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
            ema.to(device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"Resumed from {args.resume}: epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}")

    wandb = maybe_init_wandb(args)
    config = vars(args).copy()
    config["model_config"] = model.config

    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        num_batches = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            timesteps = torch.randint(0, args.timesteps, (images.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = diffusion.p_losses(model, images, timesteps, labels, cfg_drop_prob=args.cfg_drop_prob)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            loss_sum += loss_value
            num_batches += 1
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if wandb and global_step % args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

        avg_loss = loss_sum / max(1, num_batches)
        print(f"Epoch {epoch:03d}: avg_loss={avg_loss:.6f}")

        last_path = save_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss)
        copy_to_backup(last_path, args.backup_dir)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best.pt"
            best_ema_path = save_dir / "best_ema.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss)
            save_ema_checkpoint(best_ema_path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss)
            copy_to_backup(best_path, args.backup_dir)
            copy_to_backup(best_ema_path, args.backup_dir)
            print(f"Best checkpoint saved: loss={best_loss:.6f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = save_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(epoch_path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss)
            copy_to_backup(epoch_path, args.backup_dir)

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            save_preview(model, diffusion, ema, object_map, args, device, epoch)

        if stop_training:
            print(f"Stopped at max_steps={args.max_steps}")
            break

    if wandb:
        wandb.finish()
    print(f"Training finished. best_loss={best_loss:.6f}, global_step={global_step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional DDPM for i-CLEVR.")
    parser.add_argument("--data-root", type=str, default="/content/data/iclevr")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--save-dir", type=str, default="/content/lab6_runs/baseline")
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=str, default="16")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--preview-sample-steps", type=int, default=50)
    parser.add_argument("--preview-cfg-scale", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab6-Conditional-DDPM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
