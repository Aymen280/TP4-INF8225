import torch.nn as nn
from cplx_blocks import CplxDown, CplxDoubleConv, CplxUp, CplxConv2d

class CplxUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = CplxDoubleConv(n_channels, 64)
        self.down1 = CplxDown(64, 128)
        self.down2 = CplxDown(128, 256)
        self.down3 = CplxDown(256, 512)
        self.down4 = CplxDown(512, 1024)
        self.up1 = CplxUp(1024, 512)
        self.up2 = CplxUp(512, 256)
        self.up3 = CplxUp(256, 128)
        self.up4 = CplxUp(128, 64)
        self.outc = CplxConv2d(64, n_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x