from albumentations.augmentations import functional as F
from albumentations.core import transforms_interface as TI
import cv2
import numpy as np

rng = np.random.default_rng()


class GridMask(TI.ImageOnlyTransform):
    """GridMask augmentation for image classification and object detection.

    Author: @artur.k.space

    Args:
        ratio (int or (int, int)): ratio which define "l" size of grid units
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    |  https://www.kaggle.com/haqishen/gridmask
    """

    def __init__(self, ratio=(0.4, 0.7), num_grid=3, fill_value=0, rotate=90, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)

        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.ratio = 0.5 if mode == 2 else ratio
        self.hh = None  # diagonal
        self.height, self.width = None, None

    def init_masks(self, height, width):
        self.masks = []
        self.height, self.width = height, width
        self.hh = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

        for n, n_grid in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
            self.masks.append(self.make_grid(n_grid))

    def make_grid(self, n_grid):
        assert self.hh is not None

        d_h = self.height / n_grid
        d_w = self.width / n_grid

        mask = np.ones((self.hh, self.hh), np.float32)
        r = np.random.uniform(self.ratio[0], self.ratio[1]) if isinstance(self.ratio, tuple) else self.ratio

        l_h = int(np.ceil(d_h * r))
        l_w = int(np.ceil(d_w * r))

        for i in range(-1, self.hh // int(d_h) + 1):
            s = int(d_h * i + d_h)
            t = s + l_h
            s = max(min(s, self.hh), 0)
            t = max(min(t, self.hh), 0)

            if self.mode == 2:
                mask[s:t, :] = 1 - mask[s:t]  # invert
            else:
                mask[s:t, :] = self.fill_value

        for i in range(-1, self.hh // int(d_w) + 1):
            s = int(d_w * i + d_w)
            t = s + l_w
            s = max(min(s, self.hh), 0)
            t = max(min(t, self.hh), 0)

            if self.mode == 2:
                mask[:, s:t] = 1 - mask[:, s:t]  # invert
            else:
                mask[:, s:t] = self.fill_value

        if self.mode == 1:
            mask = 1 - mask

        return mask

    def apply(self, image, **params):
        h, w = image.shape[:2]

        if self.masks is None: self.init_masks(h, w)

        mask = rng.choice(self.masks)
        rand_h = np.random.randint(self.hh - h)
        rand_w = np.random.randint(self.hh - w)
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h + h, rand_w:rand_w + w].astype(image.dtype)
        return image

    def get_transform_init_args_names(self):
        return ("ratio", "num_grid", "fill_value", "rotate", "mode")


class DynamicProb(TI.BasicTransform):
    """DynamicProb improvement

    Author: @artur.k.space

    Args:
        transform (BasicTransform): Albumentations transformer instance
        final_cnt (int): final instance calls count of the last step
        p_steps (int): progression steps number
        p (tuple(float)): min prob, max prob
    """

    def __init__(self, transform: TI.BasicTransform, final_cnt: int, p_steps: int = 5, p: tuple = (0, 0.8),
                 always_apply=False):
        super().__init__(always_apply, p=1)

        transform.always_apply = True
        self.transform = transform

        self._cnt = 0
        self._stepid = 0
        self._prob = p[0]
        self.p_steps = p_steps
        self.c_range = np.linspace(0, final_cnt, num=p_steps, dtype=np.int32)
        self.p_range = np.linspace(p[0], p[1], num=p_steps, dtype=np.float32)

    def _update_p(self):
        self._stepid += 1
        self._prob = self.p_range[self._stepid]

    def __call__(self, force_apply=False, **kwargs):
        result = kwargs

        if self._prob > np.random.rand():
            result = self.transform(**kwargs)

        self._cnt += 1

        if self.p_steps - 1 > self._stepid and self._cnt >= self.c_range[self._stepid + 1]:
            self._update_p()

        return result


class RandomThickness(TI.ImageOnlyTransform):
    """ Randomly changes line thickness in the input image

    Author: @artur.k.space

    Args:
        force (int or (int, int)): minimum/maximum erose/dilate iterations
        ksize (int): size of a kernel for the dilation

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, force=3, ksize: int = 3, always_apply=False, p=1.0):

        if isinstance(force, int):
            force = (-force, force)

        assert len(force) == 2
        assert force[0] < force[1]

        super().__init__(always_apply, p)

        self.force = force
        self.ksize = ksize

    def apply(self, image, **params):

        force_range = np.arange(self.force[0], self.force[1] + 1, step=1, dtype=np.int8)
        force = np.random.choice(force_range[force_range != 0])

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.ksize, self.ksize))

        if force > 0:
            image = cv2.dilate(image, kernel=kernel, iterations=force, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            image = cv2.erode(image, kernel=kernel, iterations=abs(force), borderType=cv2.BORDER_CONSTANT,
                              borderValue=0)

        return image

    def get_transform_init_args_names(self):
        return ("force", "ksize")