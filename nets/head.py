import torch
import torch.nn as nn
import torch.nn.functional as F


# head
class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels=6625, **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True, )
        self.out_channels = out_channels

    def forward(self, x):
        predicts = self.fc(x)
        result = F.softmax(predicts, dim=2)
        return result
