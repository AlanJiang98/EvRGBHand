#import albumentations as A
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


class EvRGBDegrader(object):
    def __init__(self, config, is_train):
        self.conifg = config
        self.is_train = is_train
        self.colorjitter_aug = [
            config['colorjitter']['p'],
            transforms.ColorJitter(
                brightness=config['colorjitter']['brightness'],
                contrast=config['colorjitter']['contrast'],
            )
        ]
        self.gaussianblur_aug = [
            config['gaussianblur']['p'],
            transforms.GaussianBlur(
                kernel_size=config['gaussianblur']['kernel_size'],
                sigma=config['gaussianblur']['sigma'],
            )
        ]

    @staticmethod
    def gaussian_noise(x, var_limit):
        if x[2].any() != 0:
            var = (torch.rand(1) * var_limit[1]-var_limit[0]) + var_limit[0]
            noise = torch.randn_like(x) * var
            return x + noise
        else:
            var = (torch.rand(1) * var_limit[1]-var_limit[0]) + var_limit[0]
            noise = torch.randn_like(x) * var
            x = torch.clip(x + noise, 0, 1.)
            x[2] = 0
            return x

    @staticmethod
    def salt_pepper_noise(x, rate):
        x_ = x.clone()
        if x_[2].any() != 0:
            noise = torch.rand_like(x_)
            flipped = noise < rate
            salted = torch.rand_like(x_) > 0.5
            peppered = ~salted
            x_[flipped & salted] = 1
            x_[flipped & peppered] = 0
            return x_
        else:
            noise = torch.rand_like(x_[0])
            flipped = noise < rate
            salted = torch.rand_like(x_[0]) > 0.5
            peppered = ~salted
            x_[0, flipped & salted] = 1
            x_[1, flipped & peppered] = 1
            return x_
        # channel = (torch.rand(1) > 0.5).sum()
        # index = (torch.rand_like(x) < rate)[0]
        # x_ = x.clone()
        # x_[channel] += index * torch.rand_like(x)[0]
        # x_[:2] = torch.clip(x_[:2], 0, 1.)
        # if x_[2].any() != 0:
        #     noise = torch.randn_like(x)[0] * (channel * 2 -1)
        #     x_[2] = torch.clip(noise * index + x_[2], -1, 1)
        # return x_

    def __call__(self, x, seed=None):
        scene = torch.zeros(3)
        if not self.is_train:
            return x, scene
        else:
            if seed is not None:
                torch.manual_seed(seed)
            if torch.rand(1) < self.colorjitter_aug[0]:
                x = self.colorjitter_aug[1](x)
                scene[0] = 1
            if torch.rand(1) < self.gaussianblur_aug[0]:
                x = self.gaussianblur_aug[1](x)
                scene[1] = 1
            if torch.rand(1) < self.conifg['gauss']['p']:
                x = EvRGBDegrader.gaussian_noise(x, var_limit=self.conifg['gauss']['var'])
            if torch.rand(1) < self.conifg['salt_pepper']['p']:
                x = EvRGBDegrader.salt_pepper_noise(x, self.conifg['salt_pepper']['rate'])
                scene[2] = 1
            return x, scene