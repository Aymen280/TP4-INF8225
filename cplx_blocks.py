import torch
import torch.nn as nn
import torch.nn.functional as F

from cplx_layers import CplxAvgPool2d, CplxBatchNorm2d, CplxConv2d, cplx_relu

class CplxDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = CplxConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = CplxBatchNorm2d(out_channels)
        self.conv2 = CplxConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = CplxBatchNorm2d(out_channels)
        
    def forward(self, x):
        x = cplx_relu(self.bn1(self.conv1(x)))
        x = cplx_relu(self.bn2(self.conv2(x)))
        return x

class CplxDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = CplxAvgPool2d(2)
        self.conv = CplxDoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class CplxUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = CplxDoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1_up = torch.complex(self.up(x1.real), self.up(x1.imag))
        # Faire correspondre la taille (padding si besoin)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)
