import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing, Lambda, BorderPad, ResizeWithPadOrCrop, Flip
)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.pipeline = Compose([LoadImage(image_only=True),
                                EnsureChannelFirst(),
                                Orientation(axcodes=self.config.orientation),
                                Flip(spatial_axis=[0, 1, 2]),
                                Spacing(
                                    pixdim=(self.config.target_iso_mm, self.config.target_iso_mm, self.config.target_iso_mm),
                                    mode="bilinear",
                                    padding_mode="border"
                                 ),
                                 ResizeWithPadOrCrop(
                                    spatial_size=(128,128,128),
                                    mode="constant",
                                    constant_values=self.config.hu_window[0]
                                 ),
                                
                                Lambda(func=lambda x: torch.clamp(x, min=self.config.hu_window[0], max=self.config.hu_window[1]))

        ])

    def process(self, file_path):
        
        # (C, X, Y, Z)
        data_tensor = self.pipeline(file_path)
        
        # (X, Y, Z)
        arr_xyz = data_tensor.squeeze(0).numpy()

        # (Z, Y, X)
        arr_zyx = arr_xyz.transpose(2, 1, 0).astype(np.float32)

        # NaNs values are clipped
        arr_zyx[~np.isfinite(arr_zyx)] = self.config.hu_window[0]

        spacing_zyx = np.array([self.config.target_iso_mm]*3, dtype=np.float32)

        return arr_zyx, spacing_zyx



