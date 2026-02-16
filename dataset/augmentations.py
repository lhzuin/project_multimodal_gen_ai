# augmentations.py
from dataclasses import dataclass
import albumentations as A

@dataclass
class AlbuAugment:
    resolution: int

    def __post_init__(self):
        r = self.resolution
        self.aug = A.Compose([
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Affine(rotate=(-8, 8), translate_percent=(0.0, 0.03), scale=(0.95, 1.05), p=0.6),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),

            # Newer Albumentations API (fixes your warnings too)
            A.ImageCompression(quality_range=(40, 95), p=0.35),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(r // 12, r // 6),
                hole_width_range=(r // 12, r // 6),
                p=0.35,
            ),
        ])

    def __call__(self, np_img):
        return self.aug(image=np_img)["image"]

def make_albu_augment(resolution: int):
    return AlbuAugment(resolution)