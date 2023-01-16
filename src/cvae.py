# Conditional Variational Autoencoder (non-convolutional) (bi-level output) (implemented in PyTorch)

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CVAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.input_size = 28 * 28
        self.label_size = 10
        self.embed_size = 2
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + self.label_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_size + self.label_size, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size),
            nn.Sigmoid(),
        )
        
        self.embed_mean = nn.Linear(32, self.embed_size)
        self.embed_log_variance = nn.Linear(32, self.embed_size)
        
        
    def forward(self, input, label):
        
        input = input.view(-1, self.input_size) # flatten input
        label = F.one_hot(label, num_classes=10)        
        
        x = torch.cat((input, label), dim=-1) # Concatenate the input and conditional label.
        x = self.encoder(x)
        z_mean = self.embed_mean(x)
        z_log_variance = self.embed_log_variance(x)
        z = z_mean
        
        # Sample from the posterior when training...
        
        if self.training:
            epsilon = torch.randn_like(z_mean)
            z = z_mean + torch.exp(z_log_variance * 0.5) * epsilon

        x = torch.cat((z, label), dim=-1) # Concatenate the latent code and conditional label.
        reconstruction = self.decoder(x).view(-1, 1, 28, 28)
        
        if self.training:
            return (reconstruction, z_mean, z_log_variance)
        
        return reconstruction
    

# Loss function 

def cvae_loss(input, reconstruction, z_mean, z_log_variance):

    # Reconstruction error

    reconstruction_loss = F.binary_cross_entropy(reconstruction, input, reduction='sum')
    kl_loss = -0.5 * (1 + z_log_variance - z_mean ** 2 - z_log_variance.exp()).sum()

    return reconstruction_loss + kl_loss
