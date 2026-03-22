# HeavyTrainTransform-> https://github.com/facebookresearch/vicreg/blob/main/augmentations.py#L37


import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFilter
import random

from src.utils.enums import Augmentation, TrainingMode

def get_augmentations(transform_size: int, aug_mode: str = None, training_mode: TrainingMode = None):
    two_views = False
    train_transform = None
    
    if training_mode and aug_mode:
        two_views = False if training_mode == TrainingMode.SUPERVISED else True
        mode = Augmentation(aug_mode)
        
        if mode == Augmentation.NORMAL:
            train_transform = NormalTrainTransform(transform_size=transform_size, two_views=two_views)
        elif mode == Augmentation.OCT_CLASSIFIER:
            train_transform = OCTClassifierTrainTransform(transform_size=transform_size, two_views=two_views)
        elif mode == Augmentation.HEAVY:
            train_transform = HeavyTrainTransform(transform_size=transform_size, two_views=two_views)
        else:
            raise ValueError(f"Unsupported augmentation mode: {aug_mode}")

    test_transform = TestTransform(transform_size=transform_size, two_views=False)
    return train_transform, test_transform

class VolumeTransform:
    """
    Applies 2D transforms consistently across a 3D Volume [C, D, H, W].
    Captures RNG state to ensure every slice gets the EXACT same transformation.
    Depth augmentations are ignore. Volume transformation at slice level, not interslice level.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, volume):
        # Expected input: [C, D, H, W]
        
        # Get current random state
        state = torch.get_rng_state()

        slices = torch.unbind(volume, dim=1) # List of [C, H, W]
        processed_slices = []

        for s in slices:
            # Revert to the same state for every slice
            torch.set_rng_state(state)

            # ToPIL -> Transform -> ToTensor
            # Note: transform should include ToTensor() at the end
            processed_slices.append(self.transform(s))
        
        return torch.stack(processed_slices, dim=1)

class NormalTrainTransform(object):
    def __init__(self, transform_size=224, two_views: bool = False):
        self.two_views = two_views
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                transform_size,
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.225]), 
        ])
        self.volume_transform = VolumeTransform(self.base_transform)
    
    def __call__(self, sample):
        if self.two_views:
            return self.volume_transform(sample), self.volume_transform(sample)
        return self.volume_transform(sample)
    
class OCTClassifierTrainTransform(object):
    def __init__(self, transform_size=224, rotation=15, two_views: bool = False):
        self.two_views = two_views
        # We process 1-channel grayscale directly to match your pretrained ViT
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((transform_size, transform_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
            transforms.RandomRotation(degrees=rotation),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.225]) # Grayscale stats
        ])
        self.volume_transform = VolumeTransform(self.base_transform)
    
    def __call__(self, sample):
        if self.two_views:
            return self.volume_transform(sample), self.volume_transform(sample)
        return self.volume_transform(sample)

class HeavyTrainTransform(object):
    def __init__(self, transform_size=224, two_views=False):
        self.two_views = two_views
        
        def get_base(p_blur, p_solar):
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(transform_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                # ColorJitter and Grayscale work on PIL images
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                GaussianBlur(p=p_blur),
                Solarization(p=p_solar),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.225])
            ])

        self.trans1 = VolumeTransform(get_base(1.0, 0.0))
        self.trans2 = VolumeTransform(get_base(0.1, 0.2))

    def __call__(self, sample):
        if self.two_views:
            return self.trans1(sample), self.trans2(sample)
        return self.trans1(sample)

class TestTransform(object):
    def __init__(self, transform_size=224, two_views=False):
        self.two_views = two_views
        self.transform = VolumeTransform(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((transform_size, transform_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.225])
        ]))

    def __call__(self, sample):
        if self.two_views:
            return self.transform(sample), self.transform(sample)
        return self.transform(sample)

# Helper classes for PIL Filtering
class GaussianBlur(object):
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        return img

class Solarization(object):
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


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