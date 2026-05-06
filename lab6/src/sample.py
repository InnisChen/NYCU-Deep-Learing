from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from .dataset import ConditionDataset, labels_batch_to_multihot, load_object_map
from .diffusion import GaussianDiffusion
from .models import ConditionalUNet
from .utils import ensure_dir, get_device, load_checkpoint, save_tensor_grid, save_tensor_image, set_seed


def output_image_name(index: int) -> str:
    return f"{index}.png"


def parse_int_tuple(value):
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def checkpoint_config(ckpt, args):
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model_config", {})
    return SimpleNamespace(
        image_size=int(model_cfg.get("image_size", cfg.get("image_size", args.image_size))),
        base_channels=int(model_cfg.get("base_channels", cfg.get("base_channels", args.base_channels))),
        channel_mults=model_cfg.get("channel_mults", cfg.get("channel_mults", args.channel_mults)),
        num_res_blocks=int(model_cfg.get("num_res_blocks", cfg.get("num_res_blocks", args.num_res_blocks))),
        attention_resolutions=model_cfg.get("attention_resolutions", cfg.get("attention_resolutions", args.attention_resolutions)),
        dropout=float(model_cfg.get("dropout", cfg.get("dropout", args.dropout))),
        timesteps=int(cfg.get("timesteps", args.timesteps)),
        beta_schedule=str(cfg.get("beta_schedule", args.beta_schedule)),
    )


def load_model(args, device):
    ckpt = load_checkpoint(args.ckpt, device)
    cfg = checkpoint_config(ckpt, args)
    model = ConditionalUNet(
        image_size=cfg.image_size,
        num_classes=24,
        base_channels=cfg.base_channels,
        channel_mults=parse_int_tuple(cfg.channel_mults),
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=parse_int_tuple(cfg.attention_resolutions),
        dropout=cfg.dropout,
    ).to(device)
    if args.use_ema and ckpt.get("ema"):
        ema_state = ckpt["ema"]
        state = ema_state.get("shadow", ema_state)
        print("Loaded EMA weights from checkpoint.")
    else:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(timesteps=cfg.timesteps, beta_schedule=cfg.beta_schedule).to(device)
    return model, diffusion, cfg


@torch.no_grad()
def generate_split(args, split: str, model, diffusion, cfg, device) -> None:
    dataset = ConditionDataset(args.meta_dir, split=split)
    labels = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0).to(device)
    split_dir = ensure_dir(Path(args.out_dir) / split)

    if args.num_candidates <= 1:
        images = diffusion.ddim_sample(
            model,
            labels,
            image_size=cfg.image_size,
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            eta=args.eta,
        )
        for idx, image in enumerate(images):
            save_tensor_image(image, split_dir / output_image_name(idx))
        save_tensor_grid(images, Path(args.out_dir) / f"{split}_grid.png", nrow=8)
        return

    candidate_root = ensure_dir(Path(args.out_dir) / f"{split}_candidates")
    repeated_labels = labels.repeat_interleave(args.num_candidates, dim=0)
    images = diffusion.ddim_sample(
        model,
        repeated_labels,
        image_size=cfg.image_size,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
    )
    images = images.reshape(len(dataset), args.num_candidates, *images.shape[1:])
    for idx in range(len(dataset)):
        item_dir = ensure_dir(candidate_root / str(idx))
        for cand_idx in range(args.num_candidates):
            save_tensor_image(images[idx, cand_idx], item_dir / output_image_name(cand_idx))
        save_tensor_image(images[idx, 0], split_dir / output_image_name(idx))
    save_tensor_grid(images[:, 0], Path(args.out_dir) / f"{split}_grid.png", nrow=8)


@torch.no_grad()
def save_denoising_process(args, model, diffusion, cfg, device) -> None:
    object_map = load_object_map(args.meta_dir)
    labels = labels_batch_to_multihot([["red sphere", "cyan cylinder", "cyan cube"]], object_map).to(device)
    _, intermediates = diffusion.ddim_sample(
        model,
        labels,
        image_size=cfg.image_size,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
        return_intermediates=True,
        num_intermediates=args.denoise_frames,
    )
    frames = torch.cat(intermediates, dim=0)
    save_tensor_grid(frames, Path(args.out_dir) / "denoising_process.png", nrow=args.denoise_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate i-CLEVR test images with a trained DDPM.")
    parser.add_argument("--meta-dir", type=str, default="file/file")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="/content/images")
    parser.add_argument("--split", type=str, default="both", choices=["test", "new_test", "both"])
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=str, default="16")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)
    model, diffusion, cfg = load_model(args, device)
    splits = ["test", "new_test"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"Generating split: {split}")
        generate_split(args, split, model, diffusion, cfg, device)
    save_denoising_process(args, model, diffusion, cfg, device)
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
