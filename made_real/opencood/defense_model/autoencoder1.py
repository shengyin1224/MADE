import torch
import torch.nn as nn
from .unet_utils import *


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. E.g., for CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.conv1 = nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2) # 32x32 => 16x16
        self.act = act_fn()
        self.conv2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2) # 16x16 => 8x8
        self.conv4 = nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2) # 8x8 => 4x4
        self.flatten = nn.Flatten() # Image grid to single feature vector
        self.fc = nn.Linear(57344, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.act(x)
        # import ipdb; ipdb.set_trace()
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 57344),
            act_fn()
        )
        self.deconv1 = nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2) # 4x4 => 8x8
        self.act = act_fn()
        self.deconv2 = nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=0, padding=1, stride=2) # 8x8 => 16x16
        self.deconv4 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=0)
        self.deconv5 = nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=0, padding=1, stride=2) # 16x16 => 32x32
        self.tanh = nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well

    def forward(self, x):
        x = self.linear(x)
        # import ipdb; ipdb.set_trace()
        x = x.reshape(x.shape[0], -1, 7, 16)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        x = self.act(x)
        x = self.deconv4(x)
        x = self.act(x)
        x = self.deconv5(x)
        x = self.tanh(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # import ipdb; ipdb.set_trace()
        return x_hat, z


class BinaryClassifier(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 num_input_channels: int = 3):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.fc = nn.Linear(latent_dim, 2)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        output = self.fc(z)

        return output


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes*self.out_channels)

    def forward(self, x, embedding=False):
        # import ipdb;ipdb.set_trace()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if embedding:
            return F.adaptive_avg_pool2d(x5, (1, 1)).reshape(x5.size(0), -1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out.reshape(out.size(0), self.n_classes, self.out_channels, out.size(2), out.size(3))
        return out


class UNet_Double(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_classes=1, bilinear=False):
        super(UNet_Double, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down4 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes*self.out_channels)

    def forward(self, x, embedding=False):
        # import ipdb;ipdb.set_trace()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if embedding:
            return F.adaptive_avg_pool2d(x5, (1, 1)).reshape(x5.size(0), -1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out.reshape(out.size(0), self.n_classes, self.out_channels, out.size(2), out.size(3))
        return out

class UNet256(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 256 // factor)
        self.up1 = Up(256, 256 // factor, bilinear, ch_factor=1.5)
        self.up2 = Up(256, 256 // factor, bilinear, ch_factor=1.5)
        self.up3 = Up(256, 128 // factor, bilinear, ch_factor=1)
        self.up4 = Up(128, 64, bilinear, ch_factor=1)
        self.outc = OutConv(64, n_classes*self.out_channels)

    def forward(self, x, embedding=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if embedding:
            return F.adaptive_avg_pool2d(x5, (1, 1)).reshape(x5.size(0), -1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out.reshape(out.size(0), self.n_classes, self.out_channels, out.size(2), out.size(3))
        return out

class UNet128(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 128 // factor)
        self.up1 = Up(128, 128 // factor, bilinear, ch_factor=1.5)
        self.up2 = Up(128, 128 // factor, bilinear, ch_factor=1.5)
        self.up3 = Up(128, 128 // factor, bilinear, ch_factor=1.5)
        self.up4 = Up(128, 64, bilinear, ch_factor=1)
        self.outc = OutConv(64, n_classes*self.out_channels)

    def forward(self, x, embedding=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if embedding:
            return F.adaptive_avg_pool2d(x5, (1, 1)).reshape(x5.size(0), -1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out.reshape(out.size(0), self.n_classes, self.out_channels, out.size(2), out.size(3))
        return out

class UNet64(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 64 // factor)
        self.up1 = Up(64, 64 // factor, bilinear, ch_factor=1.5)
        self.up2 = Up(64, 64 // factor, bilinear, ch_factor=1.5)
        self.up3 = Up(64, 64 // factor, bilinear, ch_factor=1.5)
        self.up4 = Up(64, 64, bilinear, ch_factor=1.5)
        self.outc = OutConv(64, n_classes*self.out_channels)

    def forward(self, x, embedding=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if embedding:
            return F.adaptive_avg_pool2d(x5, (1, 1)).reshape(x5.size(0), -1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out.reshape(out.size(0), self.n_classes, self.out_channels, out.size(2), out.size(3))
        return out