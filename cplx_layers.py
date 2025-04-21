import torch
import torch.nn as nn
import torch.nn.functional as F

def cplx_relu(x):
    return torch.complex(F.relu(x.real), F.relu(x.imag))

def cplx_softmax(x, dim=1):
    return torch.complex(
        F.softmax(x.real, dim=dim),
        F.softmax(x.imag, dim=dim)
    )

def cplx_sigmoid(x):
    return torch.complex(
        torch.sigmoid(x.real),
        torch.sigmoid(x.imag)
    )

class CplxConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)
        return torch.complex(real, imag)

class CplxBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        real = self.real_bn(x.real)
        imag = self.imag_bn(x.imag)
        return torch.complex(real, imag)

class CplxAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.pool_real = nn.AvgPool2d(kernel_size, stride)
        self.pool_imag = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        return torch.complex(
            self.pool_real(x.real),
            self.pool_imag(x.imag)
        )
