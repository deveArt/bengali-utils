import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

IMG_H, IMG_W = 137, 236
LABEL_CLS = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']


def prepare_data(img_path,
                 data_dir,
                 label_df=None,
                 debug=False,
                 dbg_max_count=300):

    img_path = [img_path] if isinstance(img_path, str) else img_path
    parq_list = []

    for i_path in img_path:
        parq = pd.read_parquet(os.path.join(data_dir, i_path))
        parq_list.append(parq.iloc[:, 1:].values.astype(np.uint8).reshape(-1, IMG_H, IMG_W))

        if debug: break

    images = np.concatenate(parq_list, axis=0)
    if debug:
        images = images[:dbg_max_count]

    labels = label_df if label_df is None else label_df[LABEL_CLS].values[:images.shape[0]]

    return images, labels


class SmartCrop:
    """Crop the image by light pixels edges"""

    def __init__(self, size,
                 padding=15,
                 x_marg=10, y_marg=10,
                 mask_threshold=80,
                 noise_threshold=30,
                 clear_border_size: int = 2):

        self.size = size
        self.padding = padding
        self.x_marg = x_marg
        self.y_marg = y_marg
        self.noise_threshold = noise_threshold
        self.mask_threshold = mask_threshold
        self.cb_size = clear_border_size

    def __call__(self, image, remove_noise=True):
        img_h, img_w = image.shape[0], image.shape[1]

        border_mask = np.zeros(image.shape, dtype=np.uint8)
        border_mask[self.cb_size:-self.cb_size, self.cb_size:-self.cb_size] = 1

        image = image * border_mask

        img_mask = (image > self.mask_threshold).astype(np.int)
        rows = np.any(img_mask, axis=1)
        cols = np.any(img_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        #cropping may cut too much, so we need to add it back
        xmin = xmin - self.x_marg if (xmin > self.x_marg) else 0
        ymin = ymin - self.y_marg if (ymin > self.y_marg) else 0
        xmax = xmax + self.x_marg if (xmax < img_w - self.x_marg) else img_w
        ymax = ymax + self.y_marg if (ymax < img_h - self.y_marg) else img_h
        image = image[ymin:ymax, xmin:xmax]

        if remove_noise:
            image[image < self.noise_threshold] = 0 #remove low intensity pixels as noise

        lx, ly = xmax-xmin,ymax-ymin
        l = max(lx,ly) + self.padding

        #make sure that the aspect ratio is kept in rescaling
        image = np.pad(image, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
        return cv2.resize(image, (self.size, self.size))


class InverseMaxNorm8bit:
    """Inverse np.uint8 pixels and scale with max"""

    def __call__(self, x):
        x_inv = 255 - x
        x_inv = (x_inv*(255/x_inv.max())).astype(np.uint8)

        return x_inv


class Inverse8bit:
    """Inverse np.uint8 pixels"""
    def __call__(self, x):
        return (255 - x).astype(np.uint8)


class BengaliData(Dataset):
    """
    Bengali graphics data
    """

    def __init__(self, images, labels=None, out_img_size=128, transform=None):

        assert isinstance(images, np.ndarray)

        if labels is not None:
            assert isinstance(labels, np.ndarray)
            assert images.shape[0] == labels.shape[0]

        self.images = images
        self.labels = labels
        self.transform = transform
        self.inverse = InverseMaxNorm8bit()
        self.smart_crop = SmartCrop(out_img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.inverse(self.images[idx])
        x = self.smart_crop(x)

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = torch.tensor(x).unsqueeze(0)

        if self.labels is not None:
            y = self.labels[idx]
            return x, torch.tensor(y)
        else:
            return x