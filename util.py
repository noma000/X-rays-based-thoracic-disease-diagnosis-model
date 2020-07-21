
import numpy as np
import torch

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2 * 255
    return out.clamp_(0, 255)


def interpolation_list(latent_orgin, latent_target):
    latent_list = []
    for i in range(0,100):
        latent_list.append(latent_orgin * (1 - i/100) + latent_target * i/100)
    return latent_list

def to_image(normed_image,c=3,gpu=True):
    if c==3: # rgb
        origin_img = denorm(normed_image.cpu()).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return np.concatenate((origin_img, origin_img, origin_img), axis=2)
    else:
        return denorm(normed_image.cpu()).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

def to_image_np(normed_image):
    return np.concatenate((normed_image, normed_image, normed_image), axis=2).astype(dtype=np.uint8)