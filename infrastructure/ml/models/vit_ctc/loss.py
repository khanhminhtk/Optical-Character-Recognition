import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None, apply_softmax=False):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.apply_softmax = apply_softmax
        
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            assert len(weights) == len(losses), "Number of weights must match number of losses"
            self.weights = weights
        self.has_ctc = any(isinstance(loss, CTCLoss) for loss in losses)
    
    def forward(self, outputs, targets, **kwargs):
        total_loss = 0
        
        has_ctc_lengths = 'input_lengths' in kwargs and 'target_lengths' in kwargs
        
        for loss_fn, weight in zip(self.losses, self.weights):
            if isinstance(loss_fn, CTCLoss):
                input_lengths = kwargs.get('input_lengths')
                target_lengths = kwargs.get('target_lengths')
                
                if input_lengths is None or target_lengths is None:
                    raise ValueError("CTC loss requires 'input_lengths' and 'target_lengths' in kwargs")
                
                log_probs = F.log_softmax(outputs, dim=-1)
                loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
                total_loss += weight * loss
            else:
                if has_ctc_lengths:
                    continue
                
                if self.apply_softmax and not self.has_ctc:
                    outputs_processed = F.log_softmax(outputs, dim=-1)
                else:
                    outputs_processed = outputs
                
                if outputs_processed.dim() == 3:
                    outputs_processed = outputs_processed.mean(dim=1)
                
                loss = loss_fn(outputs_processed, targets)
                total_loss += weight * loss
        
        return total_loss


def create_ctc_loss(blank=0, reduction='mean', zero_infinity=True):
    return CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

def create_focal_loss(alpha=1.0, gamma=2.0):
    return FocalLoss(alpha=alpha, gamma=gamma)

def create_label_smoothing_loss(num_classes, smoothing=0.1):
    return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
