import re

import torch

from .bg_remove import POSITIONS, resolve_position
from .bg_remove_utils import hex_to_rgb

_RATIO_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")


def _parse_ratio(s):
    m = _RATIO_RE.match(s or "")
    if not m:
        return 1.0, 1.0
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return 1.0, 1.0
    return a, b


class RatioMask:
    """
    Generate a white rectangle of a chosen aspect ratio inside a colored
    canvas, fit-resized to the largest size that fits within the canvas
    minus per-side paddings, and placed at one of nine grid positions.

    The MASK output is 1.0 inside the rectangle and 0.0 outside (independent
    of bg_color). The IMAGE output paints the rectangle white over bg_color
    (so default '#000000' gives the classic white-on-black look).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "ratio": ("STRING", {"default": "1:1"}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "bg_color": ("STRING", {"default": "#000000"}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, width, height, ratio, position, bg_color,
                 padding_top, padding_bottom, padding_left, padding_right):
        rw, rh = _parse_ratio(ratio)
        target_aspect = rw / rh

        eff_w = max(1, width - padding_left - padding_right)
        eff_h = max(1, height - padding_top - padding_bottom)

        if target_aspect >= eff_w / eff_h:
            shape_w = eff_w
            shape_h = max(1, int(round(eff_w / target_aspect)))
        else:
            shape_h = eff_h
            shape_w = max(1, int(round(eff_h * target_aspect)))

        cy, cx = resolve_position(
            position, height, width, shape_h, shape_w,
            padding_top, padding_bottom, padding_left, padding_right,
        )

        rgb = hex_to_rgb(bg_color)
        bg_t = torch.tensor(rgb, dtype=torch.float32) / 255.0

        image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        image[..., :] = bg_t
        mask = torch.zeros((1, height, width), dtype=torch.float32)

        y0 = max(0, cy)
        x0 = max(0, cx)
        y1 = min(height, cy + shape_h)
        x1 = min(width, cx + shape_w)
        if y1 > y0 and x1 > x0:
            image[0, y0:y1, x0:x1, :] = 1.0
            mask[0, y0:y1, x0:x1] = 1.0

        return (image, mask)


NODE_CLASS_MAPPINGS = {"RatioMask": RatioMask}
NODE_DISPLAY_NAME_MAPPINGS = {"RatioMask": "Ratio Mask"}
