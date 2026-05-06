from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EmbedSequential(nn.Sequential):
    uses_skip: bool = False

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, ResBlock):
                x = module(x, emb)
            else:
                x = module(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),
        )
        self.out_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).type_as(h).chunk(2, dim=1)
        h = self.out_norm(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.reshape(b * self.num_heads, head_dim, h * w).transpose(1, 2)
        k = k.reshape(b * self.num_heads, head_dim, h * w)
        v = v.reshape(b * self.num_heads, head_dim, h * w).transpose(1, 2)
        weight = torch.bmm(q, k) * (head_dim**-0.5)
        weight = torch.softmax(weight, dim=-1)
        out = torch.bmm(weight, v)
        out = out.transpose(1, 2).reshape(b, c, h * w)
        out = self.proj(out).reshape(b, c, h, w)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def _parse_int_tuple(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: int = 24,
        base_channels: int = 64,
        channel_mults: str | Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: str | Sequence[int] = (16,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channel_mults = _parse_int_tuple(channel_mults)
        attention_resolutions = set(_parse_int_tuple(attention_resolutions))
        emb_dim = base_channels * 4

        self.config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_classes": num_classes,
            "base_channels": base_channels,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": sorted(attention_resolutions),
            "dropout": dropout,
        }

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, emb_dim, dropout=dropout)]
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.down_blocks.append(EmbedSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(EmbedSequential(Downsample(ch)))
                resolution //= 2
                skip_channels.append(ch)

        self.mid = EmbedSequential(
            ResBlock(ch, ch, emb_dim, dropout=dropout),
            AttentionBlock(ch),
            ResBlock(ch, ch, emb_dim, dropout=dropout),
        )

        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = EmbedSequential(
                    ResBlock(ch + skip_ch, out_ch, emb_dim, dropout=dropout),
                    *( [AttentionBlock(out_ch)] if resolution in attention_resolutions else [] ),
                )
                block.uses_skip = True
                self.up_blocks.append(block)
                ch = out_ch
            if level != 0:
                self.up_blocks.append(EmbedSequential(Upsample(ch)))
                resolution *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timesteps) + self.label_mlp(labels.float())
        h = self.input_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, emb)
            skips.append(h)
        h = self.mid(h, emb)
        for block in self.up_blocks:
            if getattr(block, "uses_skip", False):
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
            h = block(h, emb)
        return self.out(h)

