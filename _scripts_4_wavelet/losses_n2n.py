import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """조직 경계 보존"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        return (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)) / 2.0


class EdgeLoss(nn.Module):
    """Sobel edge-based loss - CT 특화"""
    def __init__(self):
        super().__init__()
        # Sobel 필터
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)


class WeightedLoss(nn.Module):
    """NS-N2N: Weighted MSE on matched regions"""
    def __init__(self):
        super().__init__()

    def forward(self, output, target, weight):
        """
        output: [B, 1, H, W]
        target: [B, 1, H, W]
        weight: [B, 1, H, W] - matched region mask
        """
        # Weighted MSE
        diff_sq = (output - target) ** 2
        weighted_sum = (diff_sq * weight).sum()
        weight_sum = weight.sum()
        
        # Matched region이 너무 적으면 전체 사용
        if weight_sum < 100:
            loss = diff_sq.mean()
            matched_ratio = 0.0
        else:
            loss = weighted_sum / weight_sum
            matched_ratio = (weight_sum / weight.numel()).item()
        
        return loss, {
            'mse': loss.item(),
            'total': loss.item(),
            'matched_ratio': matched_ratio
        }