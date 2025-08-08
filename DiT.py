import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, discount_factor, in_channels, out_channels, kernel_size, stride, padding, dim):
        super().__init__()
        self.discount_factor = discount_factor
        self.fc = nn.Linear(in_channels, out_channels)  # Fully connected layer for X_in
        self.dw_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,  # out_channels must be a multiple of in_channels
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels  # This is the key for depthwise convolution
            )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv = nn.Linear(in_channels, out_channels)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(out_channels, in_channels)

    def forward(self, x, prior):
        prior_transformed = self.fc(prior)
        out = self.dw_conv(x)

        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.pw_conv(out)

        out = out + prior_transformed

        out = self.act(out)
        out = self.pw_conv2(out)

        out = out.transpose(1, 2)
        out = out * self.discount_factor
        out = out + x
        return out

class Vocos2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, dim, num_layers, discount_factor, n_fft, hop_length):
        super(Vocos2D, self).__init__()
        
        self.pw_conv = nn.Conv2d(in_features, in_features, kernel_size=1, groups=1)
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.conv_blocks = nn.ModuleList([
            ConvNeXtBlock(discount_factor, in_features, out_features, kernel_size, stride, padding, dim)
            for i in range(num_layers)
        ])
        
        self.upsample_conv = nn.ConvTranspose2d(in_features, (n_fft // 2 + 1) * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x):
        prior = x
        x = self.pw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        # Apply ConvNeXt blocks manually to pass both x and prior
        for block in self.conv_blocks:
            x = block(x, prior)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        mag_phase = self.upsample_conv(x)
        log_mag, phase = mag_phase.chunk(2, dim=1)
        
        magnitude = torch.exp(log_mag)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complex_spec = torch.complex(real, imag)
        
        y = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False)
        return y
        

class LinearSpectrogramEstimator(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, dim, num_layers, discount_factor, n_fft, hop_length):
        super(LinearSpectrogramEstimator, self).__init__()
        self.patch_embed = nn.Conv2d(in_features, out_features, kernel_size=1, groups=1)
        self.patch_linear = nn.Linear(out_features, out_features)
        self.time_embed = nn.Linear(1, out_features)
        self.sin_embed = nn.Linear(1, out_features)
        self.cond_embed = nn.Sequential(
            nn.Linear(1, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features), 
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 1, out_features))
        self.norm = nn.LayerNorm(out_features, eps=1e-6)
        self.conv_blocks = nn.ModuleList([
            ConvNeXtBlock(discount_factor, out_features, out_features, kernel_size, stride, padding, dim)
            for i in range(num_layers)
        ])

    def forward(self, x, t, c):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x + self.pos_embed
        x = self.patch_linear(x)
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        return x