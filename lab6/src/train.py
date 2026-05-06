from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import ConditionDataset, IClevrDataset, labels_batch_to_multihot, load_object_map
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


def save_ema_checkpoint(path, model, ema, epoch, global_step, config, best_loss, metrics=None):
    ema.store(model)
    ema.copy_to(model)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "config": config,
            "best_loss": best_loss,
            "metrics": metrics or {},
        },
        path,
    )
    ema.restore(model)


def load_evaluator_for_training(meta_dir: str | Path, device: torch.device):
    if device.type != "cuda":
        print("Evaluator validation disabled: provided evaluator requires CUDA.")
        return None
    meta_dir = Path(meta_dir).resolve()
    evaluator_path = meta_dir / "evaluator.py"
    old_cwd = Path.cwd()
    try:
        os.chdir(meta_dir)
        spec = importlib.util.spec_from_file_location("lab6_evaluator_train", evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import evaluator from {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evaluator = module.evaluation_model()
        print("Evaluator loaded for validation.")
        return evaluator
    except Exception as exc:
        print(f"Evaluator validation disabled: {exc}")
        return None
    finally:
        os.chdir(old_cwd)


def load_validation_labels(meta_dir: str | Path, device: torch.device):
    test_dataset = ConditionDataset(meta_dir, split="test")
    new_test_dataset = ConditionDataset(meta_dir, split="new_test")
    test_labels = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))], dim=0).to(device)
    new_test_labels = torch.stack([new_test_dataset[i][0] for i in range(len(new_test_dataset))], dim=0).to(device)
    return test_labels, new_test_labels


@torch.no_grad()
def run_evaluator_validation(model, diffusion, ema, evaluator, test_labels, new_test_labels, args, device):
    ema.store(model)
    ema.copy_to(model)
    model.eval()
    try:
        test_images = diffusion.ddim_sample(
            model,
            test_labels,
            image_size=args.image_size,
            sample_steps=args.val_sample_steps,
            cfg_scale=args.val_cfg_scale,
            eta=args.val_eta,
        )
        new_test_images = diffusion.ddim_sample(
            model,
            new_test_labels,
            image_size=args.image_size,
            sample_steps=args.val_sample_steps,
            cfg_scale=args.val_cfg_scale,
            eta=args.val_eta,
        )
        test_acc = float(evaluator.eval(test_images, test_labels))
        new_test_acc = float(evaluator.eval(new_test_images, new_test_labels))
    finally:
        ema.restore(model)
        model.train()
    avg_acc = (test_acc + new_test_acc) / 2.0
    return test_acc, new_test_acc, avg_acc


def is_backup_epoch(args, epoch: int, stop_training: bool) -> bool:
    if not args.backup_dir or args.backup_every <= 0:
        return False
    return stop_training or epoch == args.epochs or epoch % args.backup_every == 0


def is_checkpoint_epoch(args, epoch: int, stop_training: bool) -> bool:
    if stop_training or epoch == args.epochs:
        return True
    return args.save_every > 0 and epoch % args.save_every == 0


def is_validation_epoch(args, epoch: int, stop_training: bool) -> bool:
    if args.val_every <= 0:
        return False
    return stop_training or epoch == args.epochs or epoch % args.val_every == 0


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
    best_acc = -math.inf
    last_val_metrics = {}
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
        metrics = ckpt.get("metrics", {})
        best_acc = float(metrics.get("best_acc", best_acc))
        last_val_metrics = metrics.get("last_validation", {})
        print(f"Resumed from {args.resume}: epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}")

    config = vars(args).copy()
    config["model_config"] = model.config
    evaluator = load_evaluator_for_training(args.meta_dir, device) if args.val_every > 0 else None
    val_labels = load_validation_labels(args.meta_dir, device) if evaluator is not None else None

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
                loss = diffusion.p_losses(
                    model,
                    images,
                    timesteps,
                    labels,
                    cfg_drop_prob=args.cfg_drop_prob,
                    loss_type=args.loss_type,
                )
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

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

        avg_loss = loss_sum / max(1, num_batches)
        print(f"Epoch {epoch:03d}: avg_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Best loss updated in memory: {best_loss:.6f}")

        if evaluator is not None and val_labels is not None and is_validation_epoch(args, epoch, stop_training):
            test_acc, new_test_acc, avg_acc = run_evaluator_validation(
                model,
                diffusion,
                ema,
                evaluator,
                val_labels[0],
                val_labels[1],
                args,
                device,
            )
            last_val_metrics = {
                "epoch": epoch,
                "test_acc": test_acc,
                "new_test_acc": new_test_acc,
                "avg_acc": avg_acc,
            }
            print(f"Validation: test_acc={test_acc:.6f}, new_test_acc={new_test_acc:.6f}, avg_acc={avg_acc:.6f}")
            if avg_acc > best_acc:
                best_acc = avg_acc
                metrics = {"best_acc": best_acc, "last_validation": last_val_metrics}
                best_ema_path = save_dir / "best_ema.pt"
                save_ema_checkpoint(best_ema_path, model, ema, epoch, global_step, config, best_loss, metrics=metrics)
                print(f"Best EMA checkpoint updated locally by evaluator avg_acc={best_acc:.6f}: {best_ema_path}")

        checkpoint_now = is_checkpoint_epoch(args, epoch, stop_training) or is_backup_epoch(args, epoch, stop_training)
        last_path = save_dir / "last.pt"
        last_ema_path = save_dir / "last_ema.pt"
        metrics = {"best_acc": best_acc, "last_validation": last_val_metrics}
        if checkpoint_now:
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, ema, epoch, global_step, config, best_loss, metrics=metrics)
            save_ema_checkpoint(last_ema_path, model, ema, epoch, global_step, config, best_loss, metrics=metrics)
            print(f"Checkpoints saved: {last_path}, {last_ema_path}")

        if is_backup_epoch(args, epoch, stop_training):
            copy_to_backup(last_path, args.backup_dir)
            copy_to_backup(last_ema_path, args.backup_dir)
            copy_to_backup(save_dir / "best_ema.pt", args.backup_dir)
            print(f"Backed up checkpoints to Drive at epoch {epoch}.")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            save_preview(model, diffusion, ema, object_map, args, device, epoch)

        if stop_training:
            print(f"Stopped at max_steps={args.max_steps}")
            break

    print(f"Training finished. best_loss={best_loss:.6f}, best_acc={best_acc:.6f}, global_step={global_step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional DDPM for i-CLEVR.")
    parser.add_argument("--data-root", type=str, default="/content/data/iclevr")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--save-dir", type=str, default="/content/lab6_runs/baseline")
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--backup-every", type=int, default=50, help="Copy checkpoints to backup-dir every N epochs; 0 disables cloud backup.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--loss-type", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=str, default="16")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--preview-sample-steps", type=int, default=50)
    parser.add_argument("--preview-cfg-scale", type=float, default=2.0)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--val-sample-steps", type=int, default=100)
    parser.add_argument("--val-cfg-scale", type=float, default=2.0)
    parser.add_argument("--val-eta", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
