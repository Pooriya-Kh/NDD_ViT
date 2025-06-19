import torch
from torchvision.transforms import Resize, Compose, GaussianBlur, RandomChoice, RandomApply, ColorJitter

class GaussianNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        noisy_image = image + (torch.randn(image.size()) * self.std) + self.mean
        return noisy_image

class Transforms():
    def __init__(self, image_size, p=0.5):
        self.resize = Resize(size=image_size)
        
        gaussian_blur = GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
        gaussian_noise = GaussianNoise(mean=0, std=0.05)
        color_jitter_brightness = ColorJitter(brightness=0.1)
        color_jitter_contrast = ColorJitter(contrast=0.1)
        color_jitter_saturation = ColorJitter(saturation=0.1)
        
        self.random_choice = RandomChoice([
            gaussian_blur,
            gaussian_noise,
            color_jitter_brightness,
            color_jitter_contrast,
            color_jitter_saturation
        ])

        self.random_transforms =  RandomApply([self.random_choice], p=p)

    def train(self):
        return Compose([
            self.resize,
            self.random_transforms
        ])

    def eval(self):
        return Compose([self.resize])
        
        