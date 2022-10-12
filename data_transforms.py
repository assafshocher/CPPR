# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter, ImageOps
import random
import torch
import torchvision.transforms as transforms

class TwoCropsTransform:

    def __init__(self, input_size, num_groups):
        self.united_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
        ])

        self.separate_transform = transforms.Compose([
            transforms.RandomAffine(0, translate=(8/224,8/224), scale=None, shear=None, interpolation=3, fill=0, center=None),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_groups = num_groups
        self.input_size = input_size

    def __call__(self, x):
        x = self.united_transform(x)
        groups = []
        for _ in range(self.num_groups):
            groups.append(self.separate_transform(x))
        groups = torch.cat(groups, 0)
        return groups


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return ImageOps.solarize(img)
