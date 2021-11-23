import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TwoLinesLoss(nn.Module):

    def __init__(self, use_focal_loss: bool = False, use_info_gain: bool = False):
        super(TwoLinesLoss, self).__init__()
        self.use_info_gain = use_info_gain
        self.use_focal_loss = use_focal_loss

        self.sigmoid = nn.Sigmoid()

        if use_focal_loss:
            self.focal_loss = FocalLoss(gamma=3)
        else:
            self.bce_loss = nn.BCELoss()

    def forward(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor):
        """
        x1 [B, C], y1 [B]
        x2 [B, C], y2 [B] 
        """

        y1 = y1.view(-1, 1, 1, 1, 1)
        y2 = y2.view(-1, 1, 1, 1, 1)

        loss_info_gain = torch.tensor(0.0)

        x1 = self.sigmoid(x1)
        x2 = self.sigmoid(x2)

        y1 = torch.zeros_like(x1) + y1
        y2 = torch.zeros_like(x2) + y2

        if self.use_focal_loss:
            loss_l1 = self.focal_loss(x1, y1)
            loss_l2 = self.focal_loss(x2, y2)
        else:
            loss_l1 = self.bce_loss(x1, y1)
            loss_l2 = self.bce_loss(x2, y2)
        
        loss = loss_l1 + loss_l2
        
        if self.use_info_gain:
            loss_info_gain = (x1 * x1.log())/ (x2 * x2.log())
            loss_info_gain = loss_info_gain.mean()
            loss += loss_info_gain

        return loss, [loss_l1.item(), loss_l2.item(), loss_info_gain.item()]



class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()