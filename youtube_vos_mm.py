import os
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import json
import torchvision.transforms.functional as F

class YoutubeVOSMM(Dataset):
    def __init__(self, split, args):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = args.data_root
        self.split = split
        self.crop_type = args.crop_type
        self.img_size = args.img_size

        if self.crop_type == 'center':
            self.transforms = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop((args.img_size, args.img_size)),
                transforms.ToTensor()
            ])


        self.folders = glob.glob(os.path.join(self.data_root, split, '*'))
        self.nbr_frame = args.nbr_frame
        self.data_dropout = False
        self.min_alpha = args.min_alpha
        self.max_alpha = args.max_alpha
        # with open('/scratch/ahowens_root/ahowens1/panzy/mm/threshold.json', 'r') as f:
        #     self.thresholds = json.load(f)

    def random_crop(self, img_list):
        resizes = transforms.Resize(self.img_size)
        to_tensor = transforms.ToTensor()
        i, j, h, w = transforms.RandomCrop.get_params(
            resizes(img_list[0]), output_size=(self.img_size, self.img_size))
        return [to_tensor(F.crop(resizes(img), i, j, h, w)) for img in img_list]

    def __getitem__(self, index):
        folder = self.folders[index]
        folder_name = os.path.basename(folder)

        # threshold = self.thresholds[self.split][folder_name]
        # max_alpha = self.max_alpha if self.max_alpha < threshold else threshold
        all_paths = sorted(glob.glob(os.path.join(folder, '*')))
        if self.data_dropout:
            image_paths = [all_paths[0]] + sorted(random.sample(all_paths[1:], self.nbr_frame - 1))
        else:
            image_paths = all_paths[:self.nbr_frame]
        images = []
        for path in image_paths:
            images.append(Image.open(path))
        
        if self.crop_type == 'center':
            images_ = [self.transforms(img_) for img_ in images]
        elif self.crop_type == 'random':
            images_ = self.random_crop(images)
        else:
            raise NotImplementedError


        return images_, ((self.max_alpha - self.min_alpha) * torch.rand(1) + self.min_alpha).squeeze()


    def __len__(self):
        return len(self.folders)


def get_loader(split, args, shuffle, test_mode=None):
    dataset = YoutubeVOSMM(split, args)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=4, pin_memory=True)


if __name__ == "__main__":

    import argparse
    args = argparse.Namespace()
    args.batch_size = 1
    args.num_workers = 4
    args.data_root = '/n/owens-data1/mnt/big2/data/panzy/youtube-vos-mm'
    args.nbr_frame = 2
    args.data_dropout = False
    args.img_size = 256
    args.min_alpha = 2
    args.max_alpha = 50
    args.crop_type = 'random'
    test_loader = get_loader('test', args, shuffle=False)

    images, alphas = next(iter(test_loader))
    print(len(images), images[0].shape, alphas.shape)
    # # dataset = YoutubeVOS("/n/owens-data1/mnt/big2/data/public/youtube-vos-2019", 'validation')
    # test_loader = get_loader('test', 256, '/n/owens-data1/mnt/big2/data/public/youtube-vos-2019', 4, 2, shuffle=False, num_workers=4)
    # # dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=32, pin_memory=True)
    # from tqdm import tqdm
    # for images, gt_image in tqdm(test_loader):
    #     pass
    # print(images[0].shape, len(images))
    # import cv2
    # for i in range(len(images)):
    #     for j in range(images[0].shape[0]):
    #         cv2.imwrite('/home/panzy/frame/frame'+str(j)+str(i)+'.jpg', 255*images[i][j].squeeze().permute(1,2,0).numpy())

    # # images, gt = next(iter(dataloader))
    # # import cv2
    # # cv2.imwrite('/home/panzy/FLAVR/img1.jpg', 255*images[0][0].squeeze().permute(1,2,0).numpy())
    # # cv2.imwrite('/home/panzy/FLAVR/img2.jpg', 255*images[1][0].squeeze().permute(1,2,0).numpy())
    # # cv2.imwrite('/home/panzy/FLAVR/img3.jpg', 255*images[2][0].squeeze().permute(1,2,0).numpy())
    # # cv2.imwrite('/home/panzy/FLAVR/img4.jpg', 255*images[3][0].squeeze().permute(1,2,0).numpy())
    # # # cv2.imwrite('/home/panzy/FLAVR/img3.jpg', 255*images[2][0].squeeze().permute(1,2,0).numpy())
    # # cv2.imwrite('/home/panzy/FLAVR/gt.jpg', 255*gt[0][0].squeeze().permute(1,2,0).numpy())
    # # print(len(images), images[0].shape, len(gt), gt[0].shape)