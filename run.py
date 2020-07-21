

from model import Generator
from model import Discriminator
import cv2
from util import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def normalizatrion(input):
    input = input.astype(float) / 255.
    input = (input - 0.5) * 2
    return input

def run(config):
    # Load options
    file_path = config.file_path
    g_path = config.g_path
    d_path = config.d_path

    # Initial values
    g_reapeat_num = 6
    d_reapeat_num = 6
    c_dim = 2
    g_conv_dim = 64
    image_size = 256
    d_conv_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make generator netwrok and disciminator
    G = Generator(g_conv_dim, c_dim, g_reapeat_num).to(device)
    D = Discriminator(image_size, d_conv_dim, c_dim, d_reapeat_num).to(device)

    # Load Networks
    G.load(g_path)
    D.load(d_path)

    # Load X-ray image
    image = Image.open(file_path)
    w, h = image.size
    image = np.array(image.getdata()).reshape(w,h,3)
    image = image.mean(axis=2).reshape(1, 1, w, h)
    image = normalizatrion(image)

    input_image = torch.from_numpy(image).type(torch.FloatTensor).to(device)

    # Generate target vector
    Target_normal = torch.tensor(np.array([[[0],[1]]]))
    Target_normal = Target_normal.type(torch.FloatTensor).to(device)

    # Synthesis to normal image
    Normal_image = G(input_image, Target_normal)

    # Make lesionmap
    # calculate difference input and output
    lesion_map = input_image - Normal_image[0]
    lesion_map = to_image(lesion_map[0], c=1)
    lesion_map = cv2.applyColorMap(lesion_map, cv2.COLORMAP_JET)

    input_img = to_image(input_image[0], c=3)
    normal_img = to_image(Normal_image[0], c=3)
    synthesis_img = (input_image * 0.7 + lesion_map * 0.3).astype(np.uint8)

    subject_image = cv2.hconcat([input_img, lesion_map, normal_img, synthesis_img])
    plt.imshow(subject_image)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default="./data/x-ray_image.jpg", help = 'Load file')
    parser.add_argument('--g_path', type=str, default="./model/save/20500-G.ckpt", help = 'Generator model path')
    parser.add_argument('--d_path', type=str, default="./model/save/20500-D.ckpt", help = 'Discriminator model path')

    config = parser.parse_args()
    print(config)
    run(config)