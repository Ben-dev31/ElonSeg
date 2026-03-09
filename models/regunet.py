
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        return self.block(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)



class Up(nn.Module):
    def __init__(self, in_channels_from_deeper, skip_channels, out_channels):
        super().__init__()
        # ConvTranspose2d upsamples the feature map from the deeper layer
        self.up = nn.ConvTranspose2d(in_channels_from_deeper, in_channels_from_deeper // 2, 2, stride=2)
        # The DoubleConv will take the concatenated input from upsampled and skip connections
        self.conv = DoubleConv((in_channels_from_deeper // 2) + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ajustement spatial si nécessaire
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_filters=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, n_filters)
        self.down1 = Down(n_filters, n_filters * 2)
        self.down2 = Down(n_filters * 2, n_filters * 4)
        self.down3 = Down(n_filters * 4, n_filters * 8)
        self.down4 = Down(n_filters * 8, n_filters * 8)

        # Up(in_channels_from_deeper, skip_channels, out_channels)

        self.up1 = Up(n_filters*8, n_filters*8, n_filters*4) # Deeper x5 (512), Skip x4 (512), Out (256)
        self.up2 = Up(n_filters*4, n_filters*4, n_filters*2) # Deeper up1 output (256), Skip x3 (256), Out (128)
        self.up3 = Up(n_filters*2, n_filters*2, n_filters)  # Deeper up2 output (128), Skip x2 (128), Out (64)
        self.up4 = Up(n_filters, n_filters, n_filters)    # Deeper up3 output (64), Skip x1 (64), Out (64)

        self.outc = nn.Conv2d(n_filters, 1, kernel_size=1)

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

        return self.outc(x)

class UNetRegressor(UNet):
    def __init__(self, in_channels=3, n_filters=32):
        super().__init__(in_channels, n_filters)

        # Initialisation biais positif (évite collapse à zéro)
        nn.init.constant_(self.outc.bias, 0.1)

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
        return F.softplus(x)


class UNetClassifier(UNet):
    def __init__(self, in_channels=3, n_filters=32):
        super().__init__(in_channels, n_filters)

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