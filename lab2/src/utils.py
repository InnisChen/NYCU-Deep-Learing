import torch


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
