import torch
import torch.nn.functional as F


def dice_score(pred_logits, target, threshold=0.5):
    """
    pred_logits : (B, 1, H, W)  模型原始輸出（未經 sigmoid）
    target      : (B, 1, H, W)  binary mask（值為 0 或 1）
    回傳 batch 內的平均 Dice Score
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_size    = pred.sum(dim=(1, 2, 3))
    gt_size      = target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection) / (pred_size + gt_size + 1e-8)
    return dice.mean().item()


def dice_components(pred_logits, target, threshold=0.5):
    """
    回傳 (intersection, pred_size, gt_size) 的 batch 總和，
    供外部累積後計算 global Dice（與 Kaggle 評分方式一致）：
        global_dice = 2 * total_intersection / (total_pred + total_gt)
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum().item()
    pred_size    = pred.sum().item()
    gt_size      = target.sum().item()
    return intersection, pred_size, gt_size


def dice_loss(pred_logits, target):
    """
    Soft Dice loss（可微分，適合 training）
    pred_logits : (B, 1, H, W)  未經 sigmoid
    target      : (B, 1, H, W)  float, 0 or 1
    """
    pred = torch.sigmoid(pred_logits.float())   # float32：避免 AMP float16 加總溢位
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_size    = pred.sum(dim=(1, 2, 3))
    gt_size      = target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (pred_size + gt_size + 1.0)
    return 1.0 - dice.mean()


def bce_dice_loss(pred_logits, target, bce_weight=0.5):
    """BCE + Dice 組合 loss，預設各占 50%"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dl  = dice_loss(pred_logits, target)
    return bce_weight * bce + (1.0 - bce_weight) * dl
