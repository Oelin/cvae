from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


dataset = torchvision.datasets.MNIST(root="data-mnist", download=True, transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.MNIST(root="data-mnist", train=False, transform=torchvision.transforms.ToTensor())

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)


autoencoder_4 = CVAE()
loss_fn = cvae_loss
optimizer = torch.optim.Adam(autoencoder_4.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    
    for b, (input, label) in enumerate(loader):
        
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_variance = autoencoder_4(input, label)
        
        loss = loss_fn(input, reconstruction, z_mean, z_log_variance)
        loss.backward()
        optimizer.step()
        
        if (b % 160) == 0:

            print("Epoch {}: Batch [{}/{}] Loss: {}".format(
                epoch,
                b + 1,
                len(loader),
                loss.item() / input.size(0)
            ))
