import torch
import torch.nn as nn


class loss_CE(nn.Module):
    def __init__(self):
        super(loss_CE, self).__init__()
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred, target):
        output = self.CELoss(pred, target.cuda())

        return output


class loss_Discriminator(nn.Module):
    def __init__(self):
        super(loss_Discriminator, self).__init__()
        self.BCELoss = nn.BCELoss().cuda()

    def forward(self, pred, target):
        output = self.BCELoss(pred, target.cuda())

        return output
