import os
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import json

class DeepMagTest(Dataset):
    def __init__(self, img_size, data_root):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.folders = ['frameA', 'frameB', 'amplified']

        with open(os.path.join(data_root, 'data.json'), 'r') as f:
            self.data_dict = json.load(f)

        self.filenames = list(self.data_dict.keys())
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        images = []
        fn = self.filenames[index]
        for folder in self.folders:
            path = os.path.join(self.data_root, folder, fn)
            images.append(Image.open(path))
            
        images_ = [self.transforms(img_) for img_ in images]

        return images_[0:2], images_[-1], fn, self.data_dict[fn]['a'], self.data_dict[fn]['x']

    def __len__(self):
        return len(self.filenames)

def get_loader(img_size, data_root, num_workers):
    dataset = DeepMagTest(img_size, data_root)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataloader = get_loader(256, '/n/owens-data1/mnt/big2/data/panzy/deep_mag_test', 4)
    images, gt, a, flow = next(iter(dataloader))
    import cv2
    print(len(images), images[0].shape, len(gt), gt[0].shape, a, flow)
    cv2.imwrite('/home/panzy/img1.jpg', 255*images[0][0].squeeze().permute(1,2,0).flip([-1]).numpy())
    cv2.imwrite('/home/panzy/img2.jpg', 255*images[1][0].squeeze().permute(1,2,0).flip([-1]).numpy())
    cv2.imwrite('/home/panzy/gt1.jpg', 255*gt[0].squeeze().permute(1,2,0).flip([-1]).numpy())
    # 2 torch.Size([1, 3, 256, 256]) 1 torch.Size([3, 256, 256]) tensor([10.3070], dtype=torch.float64) tensor([0.9717], dtype=torch.float64)