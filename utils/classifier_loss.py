import torch
import torch.nn as nn
import torch.nn.functional as F

# CrossEntropyLoss = softmax + log + NLLLoss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input_tensor, target_tensor):
        assert input_tensor.shape[0] == target_tensor.shape[0]
        
        prob = F.softmax(input_tensor, dim = -1)
        log_prob = torch.log(prob + 1e-8)
        
        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction=self.reduction
        )

        return loss


class FocalLoss_1(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss_1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
 
    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=2).float()
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
 
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
