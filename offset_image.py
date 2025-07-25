import torch
import torch.nn.functional as F
from PIL import ImageFilter
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

class offset_image.py:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "mask_thickness": ("INT", {"default": 10, "min": 1, "max": 512, "step": 1}),
                "mask_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("offset_image", "seam_mask",)
    FUNCTION = "offset_image"
    CATEGORY = "image/transform"

    def offset_image(self, image, offset_x, offset_y, mask_thickness, mask_blur):
        b, h, w, c = image.shape
        image_offset = torch.roll(image, shifts=(offset_y, offset_x), dims=(1, 2))

        # Create mask
        mask = torch.zeros((h, w), dtype=torch.float32)

        if offset_x != 0:
            x_start = max(0, w - abs(offset_x)) if offset_x < 0 else 0
            x_end = min(w, abs(offset_x)) if offset_x > 0 else w
            mask[:, x_start:x_start + mask_thickness] = 1.0 if offset_x > 0 else mask[:, x_end - mask_thickness:x_end] = 1.0

        if offset_y != 0:
            y_start = max(0, h - abs(offset_y)) if offset_y < 0 else 0
            y_end = min(h, abs(offset_y)) if offset_y > 0 else h
            mask[y_start:y_start + mask_thickness, :] = 1.0 if offset_y > 0 else mask[y_end - mask_thickness:y_end, :] = 1.0

        # Optional blur
        if mask_blur > 0:
            pil_mask = to_pil_image(mask)
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            mask = to_tensor(pil_mask).squeeze(0)

        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)  # Convert to batch mask
        return (image_offset, mask)
