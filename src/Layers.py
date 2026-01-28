import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockEncoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        return F.relu(out + residual)

class DownSampleResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)
        self.skip_connection_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        residual = self.skip_connection_conv(residual)
        return F.relu(out + residual)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            DownSampleResidualBlock(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            DownSampleResidualBlock(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            ResidualBlockEncoder(64),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(256),
        )

    def forward(self, x):
        return self.encoder(x)

class ResidualBlockDecoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.transpose_conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.transpose_conv2(out)
        out = self.batch_norm2(out)
        out = F.relu(out + residual)
        return out

class UpSampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.residual_tranpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.transpose_conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.transpose_conv2(out)
        out = self.batch_norm2(out)
        residual = self.residual_tranpose_conv(residual)
        out = F.relu(out + residual)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            UpSampleResidualBlock(16, 8),
            ResidualBlockDecoder(8),
            ResidualBlockDecoder(8),
            ResidualBlockDecoder(8),
            ResidualBlockDecoder(8),
            UpSampleResidualBlock(8, 4),
            ResidualBlockDecoder(4),
            ResidualBlockDecoder(4),
            ResidualBlockDecoder(4),
            ResidualBlockDecoder(4),
            UpSampleResidualBlock(4, 3),
            ResidualBlockDecoder(3),
            ResidualBlockDecoder(3),
            ResidualBlockDecoder(3),
            ResidualBlockDecoder(3),
            nn.Sigmoid(),
        )

        self.variance_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Flatten()
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.reshape(-1, 16, 4, 4)
        out = self.decoder(x)
        return out, self.flatten(x), self.variance_block(x)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)