from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

class RSNA_Pneumonia(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        # 5 attributs exist
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.dataset = []
        #self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_images = len(self.dataset)


    def preprocess(self):
        """Preprocess the CheXpert attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                # 0 : Negative, 1 : Positive
                label.append(float(values[idx]))
            self.dataset.append([filename + '.jpg', label])

        print('Finished preprocessing the CheXPert dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.dataset
        #filename, label, label_statue = dataset[index]
        filename, label = dataset[index]
        # Change rgb to grey scale
        rgb_image = Image.open(os.path.join(self.image_dir, filename))
        w, h = rgb_image.size
        image = self.transform(rgb_image).mean(dim=0).reshape((1,w,h))

        return filename.split('.')[0], image, torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=256, image_size=256,
               batch_size=8, dataset=' RSNA_Pneumonia', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        print("train")
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.RandomAffine(shear=(-4,4),degrees=(-7,7)))
        transform.append(T.ColorJitter(brightness=(0.2,1)))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)

    if dataset == 'RSNA_Pneumonia':
        dataset = RSNA_Pneumonia(image_dir, attr_path, selected_attrs, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode =='train'),
                                  num_workers=num_workers)
    return data_loader