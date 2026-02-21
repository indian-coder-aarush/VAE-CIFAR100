import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockEncoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_norm1 = nn.GroupNorm(32, in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.layer_norm2(out)
        return F.relu(out + residual)

class DownSampleResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer_norm1 = nn.GroupNorm(32, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_connection_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.layer_norm2(out)
        residual = self.skip_connection_conv(residual)
        return F.relu(out + residual)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlockEncoder(64),
            DownSampleResidualBlock(64),
            ResidualBlockEncoder(64),
            DownSampleResidualBlock(64),
            ResidualBlockEncoder(64),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(16,32),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.variance_block = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(16, 32),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        h = self.backbone(x)
        z_var = self.variance_block(h).clamp(min = -6, max = 2)
        z_mu = self.encoder(h)
        z_sample = z_mu + torch.randn_like(z_mu)*(z_var/2).exp()
        return z_sample, z_mu, z_var

class ResidualBlockDecoder(nn.Module):

    def __init__(self, in_channels):
        self.in_groups = None
        for i in reversed(range(1,in_channels)):
            if in_channels % i == 0:
                self.in_groups = i
        super().__init__()
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.transpose_conv1(x)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.transpose_conv2(out)
        out = self.layer_norm2(out)
        out = F.relu(out + residual)
        return out

class UpSampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_groups, self.out_groups = None, None
        for i in reversed(range(1,in_channels)):
            if in_channels % i == 0:
                self.in_groups = i
        for i in reversed(range(1, in_channels)):
            if out_channels % i == 0:
                self.out_groups = i
        super().__init__()
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_tranpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.transpose_conv1(x)
        out = F.relu(out)
        out = self.transpose_conv2(out)
        residual = self.residual_tranpose_conv(residual)
        out = out + residual
        return out

class Reshape(nn.Module):
    def __init__(self, reshape_dim):
        super().__init__()
        if(isinstance(reshape_dim, tuple)):
            self.reshape_dim = reshape_dim
        else:
            raise ValueError("reshape dim must be tuple")

    def forward(self, x):
        return x.reshape(*self.reshape_dim)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            UpSampleResidualBlock(8, 8),
            nn.Dropout2d(0.1),
            ResidualBlockDecoder(8),
            UpSampleResidualBlock(8, 4),
            UpSampleResidualBlock(4, 3),
            ResidualBlockDecoder(3),
        )

    def forward(self, z, z_mu, z_var):
        out = self.decoder(z)
        out.clamp(min = 0, max = 1)
        return out, z_mu, self.flatten(z), z_var

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(*z)