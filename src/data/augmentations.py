# HeavyTrainTransform-> https://github.com/facebookresearch/vicreg/blob/main/augmentations.py#L37

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import PIL
from PIL import ImageFilter
import random

from src.utils.enums import Augmentation, TrainingMode

def get_augmentations(transform_size:int, aug_mode:str=None, training_mode:TrainingMode=None):

    # use two augmentations for PairwiseCELoss
    two_views = False
    train_transform = None
    if training_mode and aug_mode:
        two_views = True if training_mode in [TrainingMode.SIMSIAM, TrainingMode.CONTRASTIVE] else False
        mode = Augmentation(aug_mode)
        if mode == Augmentation.NORMAL:
            train_transform = NormalTrainTransform(transform_size=transform_size, two_views=two_views)
        elif mode == Augmentation.OCT_CLASSIFIER:
            train_transform = OCTClassifierTrainTransform(transform_size=transform_size, two_views=two_views)
        elif mode == Augmentation.HEAVY:
            train_transform = HeavyTrainTransform(transform_size=transform_size, two_views=two_views)
        
        else:
            raise ValueError(f"Unsupported augmentation mode: {aug_mode}")

    test_transform = TestTransform(transform_size=transform_size, two_views=two_views)
    return train_transform, test_transform

# # classification OCT Taha
# train_transform = transforms.Compose([transforms.RandomAffine(0,(0.05,0.05),fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
#                                       transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.ConvertImageDtype(torch.float32),
#                                       transforms.Normalize(*NORM)])

class OCTClassifierTrainTransform(object):
    def __init__(self, transform_size=224, rotation=15, two_views: bool = False):
        self.two_views = two_views

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # Ensure 3 channels for ViT/DINO compatibility
            transforms.Lambda(lambda x: x.convert('RGB')), 
            # Resize to consistent input size before geometric transforms
            transforms.Resize((transform_size, transform_size), interpolation=InterpolationMode.BILINEAR),
            # Taha's specific geometric augmentations
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05), 
                fill=0, 
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomRotation(
                degrees=rotation, 
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Convert to Tensor and apply normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225], 
                )
        ])
    
    def __call__(self, sample):
        if self.two_views:
            return self.transform(sample), self.transform(sample)
        else:
            return self.transform(sample)

class NormalTrainTransform(object):
    def __init__(self, transform_size=224, two_views: bool = False):
        self.two_views = two_views

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.convert('RGB')), # duplicate channel if grayscale -> compatibility with ImageNet
            transforms.RandomResizedCrop(
                transform_size,
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225], 
            ),
        ])
    
    def __call__(self, sample):
        if self.two_views:
            return self.transform(sample), self.transform(sample)
        else:
            return self.transform(sample)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class HeavyTrainTransform(object):
    def __init__(self, transform_size=224, two_views=False):
        self.two_views = two_views
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: x.convert('RGB')), # duplicate channel if grayscale -> compatibility with ImageNet
                transforms.RandomResizedCrop(
                    transform_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC, # mod -> bigger crop avoid background only imgs
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet
                    std=[0.229, 0.224, 0.225], 
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: x.convert('RGB')), # duplicate channel if grayscale -> compatibility with ImageNet
                transforms.RandomResizedCrop(
                    transform_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC # mod -> bigger crop avoid background only imgs
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet
                    std=[0.229, 0.224, 0.225], 
                ),
            ]
        )

    def __call__(self, sample):
        if self.two_views:
            return self.transform(sample), self.transform_prime(sample)
        else:
            return self.transform(sample)
    
class TestTransform(object):
    def __init__(self, transform_size=224, two_views=False):
            self.two_views = two_views
            # slightly larger resize to maintain aspect ratio before crop
            resize_dim = int((256 / 224) * transform_size)

            self.transform = transforms.Compose([
                transforms.ToPILImage(),                 
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.Resize(
                    resize_dim, # Dynamic resize
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(transform_size),  # Final output will be 64x64
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet
                    std=[0.229, 0.224, 0.225],  
                ),
            ])

    def __call__(self, sample):
        if self.two_views:
            return self.transform(sample), self.transform(sample)
        else:
            return self.transform(sample)


# class GaussianBlurSimSiam(object):
#     """Gaussian blur augmentation for SimSiam"""
#     def __init__(self, p=0.5, sigma=[0.1, 2.0]):
#         self.p = p
#         self.sigma = sigma

#     def __call__(self, x):
#         if random.random() < self.p:
#             sigma = random.uniform(self.sigma[0], self.sigma[1])
#             x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x

# class TrainTransformSimSiam(object):
#     """
#     Official SimSiam augmentation pipeline.
#     Generates two differently augmented views of the same image.
#     """
#     def __init__(self):
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )

#         self.base_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             GaussianBlurSimSiam(p=0.5, sigma=[0.1, 2.0]),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             self.normalize
#         ])

#     def __call__(self, sample):
#         view1 = self.base_transform(sample)
#         view2 = self.base_transform(sample)
#         return view1, view2