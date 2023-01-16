def render_reconstruction(input, label):
    
    reconstruction, z_mean, z_log_variance = autoencoder_4(input, label)
    
    reconstruction = reconstruction.detach().numpy()
    reconstruction = (reconstruction * 255).astype('uint8')
    reconstruction = reconstruction.reshape((28, 28))
    
    return Image.fromarray(reconstruction), z_mean, z_log_variance

  
def render_decoding(z, label):
    
    label = F.one_hot(torch.tensor([ label ]), num_classes=10) 
    
    x = torch.cat((z, label), dim=-1) # Concatenate the latent code and conditional label.
    reconstruction = autoencoder_4.decoder(x).view(-1, 1, 28, 28)
    reconstruction = reconstruction.detach().numpy()
    reconstruction = (reconstruction * 255).astype('uint8')
    reconstruction = reconstruction.reshape((28, 28))
    
    return Image.fromarray(reconstruction)
