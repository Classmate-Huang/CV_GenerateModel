import torch
from torch import random
import torch.nn as nn


def ConvBnRelu(channel_in, channel_out):
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, 3, stride=2, padding=1),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return conv_bn_relu


def DConvBnRelu(channel_in, channel_out):
    d_conv_bn_relu = nn.Sequential(
        nn.ConvTranspose2d(channel_in, channel_out, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return d_conv_bn_relu


class VariationAutoEncoder(nn.Module):
    
    def __init__(self, in_channel=3, img_size=512, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(in_channel, 96),
            ConvBnRelu(96,128),
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
        )
        self.decoder = nn.Sequential(
            DConvBnRelu(256, 256),
            DConvBnRelu(256, 128),
            DConvBnRelu(128,96),
            # nn.ConvTranspose2d(96, 3, 3, stride=2, padding=1, output_padding=1),
            DConvBnRelu(96, 96),
            nn.Conv2d(96,in_channel, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.latent_dim = latent_dim
        self.img_size = img_size
        original_dim = 256*(img_size//16)**2
        self.fc_mu = nn.Linear(original_dim, latent_dim)
        self.fc_var = nn.Linear(original_dim, latent_dim)
        self.fc_recover = nn.Linear(latent_dim, original_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # encode
        fea = self.encoder(x)
        fea = torch.flatten(fea, start_dim=1)
        
        # split into mu an var components of the latent Gaussian distribution
        mu = self.fc_mu(fea)
        log_var = self.fc_var(fea)

        # get latent code
        z = self.reparameterize(mu, log_var)

        # decode
        fea = self.fc_recover(z).view(-1, 256, self.img_size//16, self.img_size//16)
        out = self.decoder(fea)

        return mu, log_var, out
    
    def sample(self, num_sample, device):

        z = torch.randn(num_sample, self.latent_dim).to(device)
        
        fea = self.fc_recover(z).view(-1, 256, self.img_size//16, self.img_size//16)
        out = self.decoder(fea)

        return out


class FC_VAE(nn.Module):
    
    def __init__(self, in_channel, img_size=64, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        in_dim = img_size ** 2 ** in_channel
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.Linear(512, in_dim)
        )
        
        self.fc_rec = nn.Linear(latent_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
    
    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        b, c, h, w = x.shape
        
        fea = torch.flatten(x, start_dim=1)
        fea = self.encoder(fea)

        mu = self.fc_mu(fea)
        log_var = self.fc_var(fea)

        z = self.reparameter(mu, log_var)
        fea = self.fc_rec(z)

        fea = self.decoder(fea)
        out = fea.reshape(b, c, h, w)

        return mu, log_var, out

    def sample(self, num_sample, device):

        z = torch.randn(num_sample, self.latent_dim).to(device)
        
        fea = self.fc_rec(z)
        fea = self.decoder(fea)
        out = fea.reshape(num_sample, 1, 64, 64)

        return out



if __name__ == '__main__':
    x = torch.randn(3, 1, 64, 64)
    # model = VariationAutoEncoder(in_channel=1, img_size=64)
    model = FC_VAE(in_channel=1, img_size=64, latent_dim=2)
    # print(model)
    mu, logvar, y = model(x)
    print(mu.shape, logvar.shape, y.shape)
    