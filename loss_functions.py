import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import dice_coeff, focal_loss 


class RootDistanceSmoothLoss(torch.nn.Module):
    def __init__(self, alpha=5.0, beta=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        weight = 1.0 + self.alpha * target
        smooth = F.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        return (weight * smooth).mean()


class FocalLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        fl = focal_loss(logits, target, eps=self.eps)
        return fl

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, target, eps=self.eps)
        return 1.0 - dice


class SmoothL1GradLoss(nn.Module):
    def __init__(self, lambda_grad=0.05):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, pred, target):
        return self.loss_option2(pred, target, lambda_grad=self.lambda_grad)
    
    
    def gradient_loss(self, pred, target):

        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]

        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.mse_loss(dx_pred, dx_true)
        loss_y = F.mse_loss(dy_pred, dy_true)

        return loss_x + loss_y
    
    def loss_option2(self,pred, target, lambda_grad=0.3):

        smooth = F.smooth_l1_loss(pred, target)
        grad = self.gradient_loss(pred, target)

        return smooth + lambda_grad * grad

class SmoothL1GradDice(nn.Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold 

    def forward(self, pred, target):
        return self.loss_option3(pred, target)
    
    def dice_loss(self, pred, target, eps=1e-6):

        pred = pred < self.threshold
        target = target < self.threshold

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2 * intersection + eps) / (union + eps)

        return 1 - dice

    def loss_option3(self,pred, target):

        smooth = F.smooth_l1_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return 0.5 * smooth + 0.2 * grad + 0.3 * dice
    
    def gradient_loss(self, pred, target):

        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]

        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.mse_loss(dx_pred, dx_true)
        loss_y = F.mse_loss(dy_pred, dy_true)

        return loss_x + loss_y

class DiceDistanceLoss(nn.Module):
    def __init__(self, eps=1e-7, threshold=0.05):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, pred, target):
       # apply a simple minimum thresholding by subtracting the threshold
       # and clamping to keep values non-negative; this preserves gradients.
       pred_bin = torch.clamp(pred - self.threshold, min=0.0)
       target_bin = torch.clamp(target - self.threshold, min=0.0)
       return self.dice_loss(pred_bin, target_bin) 

    def dice_loss(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2 * intersection + self.eps) / (union + self.eps)

        return 1 - dice

class FocalDistanceLoss(nn.Module):
    def __init__(self, eps=1e-7, threshold=0.05):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, pred, target):
        return self.focal_loss(pred, target)

    def focal_loss(self, logits, target, alpha=0.25, gamma=2.0):
        # use clamped thresholding to create a soft binary-like map
        pred_bin = torch.clamp(logits - self.threshold, min=0.0)
        target_bin = torch.clamp(target - self.threshold, min=0.0)

        pt = torch.where(target_bin == 1, pred_bin, 1 - pred_bin)
        focal_weight = alpha * (1 - pt) ** gamma
        bce = - (target_bin * torch.log(pred_bin + self.eps) + (1 - target_bin) * torch.log(1 - pred_bin + self.eps))
        loss = focal_weight * bce
        return loss.mean()
       