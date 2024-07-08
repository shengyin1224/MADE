import torch
import torch.nn as nn


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
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3,
                      padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


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
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
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
        self.encoder = encoder_class(
            num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(
            num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class BinaryClassifier(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 num_input_channels: int = 3):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(
            num_input_channels, base_channel_size, latent_dim)
        self.fc = nn.Linear(latent_dim, 2)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        output = self.fc(z)

        return output

class Normalize(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std

def build_autoencoder(state_dict="/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/model/autoencoder_512.pth"):
    autoencoder = Autoencoder(base_channel_size=32,
                              latent_dim=512, num_input_channels=256)
    state_dict = torch.load(state_dict)
    autoencoder.load_state_dict(state_dict)
    return autoencoder

RESIDUAL_AE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_residual/model/autoencoder.pth"
def build_residual_autoencoder(checkpoint=RESIDUAL_AE, data_center=None):
    if data_center:
        autoencoder = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        autoencoder.load_state_dict(torch.load(checkpoint))
        autoencoder = nn.Sequential(
            Normalize(),
            autoencoder
        )
    else:
        autoencoder = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        autoencoder.load_state_dict(torch.load(checkpoint))
    return autoencoder

RAW_AE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_raw_features/model/autoencoder.pth"
def build_raw_autoencoder(checkpoint=RAW_AE, data_center=None):
    if data_center:
        autoencoder = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        autoencoder.load_state_dict(torch.load(checkpoint))
        autoencoder = nn.Sequential(
            Normalize(),
            autoencoder
        )
    else:
        autoencoder = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        autoencoder.load_state_dict(torch.load(checkpoint))
    return autoencoder