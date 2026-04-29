import torch
import torch.nn.functional as F

from .bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox

POSITIONS = [
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


def _resolve_position(position, canvas_h, canvas_w, asset_h, asset_w):
    v, h = position.split("-")
    if v == "top":
        cy = 0
    elif v == "bottom":
        cy = canvas_h - asset_h
    else:
        cy = (canvas_h - asset_h) // 2
    if h == "left":
        cx = 0
    elif h == "right":
        cx = canvas_w - asset_w
    else:
        cx = (canvas_w - asset_w) // 2
    return cy, cx


class BGRemoveComposeAlpha:
    """
    Remove background with BiRefNet/RMBG-2.0, then place the asset on a
    transparent canvas at a 9-grid position. Output is RGBA (4-channel IMAGE)
    plus the alpha as a MASK.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODEL_REGISTRY.keys()), {"default": "BiRefNet"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "compose"
    CATEGORY = "image/bg-remove"

    def compose(self, image, model, width, height, position, scale):
        b, _, _, _ = image.shape
        masks = predict_mask(image, model)

        out_imgs = torch.zeros((b, height, width, 4), dtype=torch.float32)
        out_masks = torch.zeros((b, height, width), dtype=torch.float32)

        for i in range(b):
            img_i = image[i]
            mask_i = masks[i]
            bbox = mask_bbox(mask_i)
            if bbox is None:
                continue
            y0, x0, y1, x1 = bbox
            asset = img_i[y0:y1, x0:x1, :]
            alpha = mask_i[y0:y1, x0:x1]

            ah, aw = asset.shape[:2]
            new_h = max(1, int(round(ah * scale)))
            new_w = max(1, int(round(aw * scale)))

            asset_chw = asset.permute(2, 0, 1).unsqueeze(0)
            asset_resized = F.interpolate(asset_chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            asset_resized = asset_resized.squeeze(0).permute(1, 2, 0)

            alpha_resized = F.interpolate(
                alpha.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

            cy, cx = _resolve_position(position, height, width, new_h, new_w)

            dst_y0 = max(0, cy)
            dst_x0 = max(0, cx)
            dst_y1 = min(height, cy + new_h)
            dst_x1 = min(width, cx + new_w)
            src_y0 = dst_y0 - cy
            src_x0 = dst_x0 - cx
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            src_x1 = src_x0 + (dst_x1 - dst_x0)

            if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                continue

            asset_crop = asset_resized[src_y0:src_y1, src_x0:src_x1, :]
            alpha_crop = alpha_resized[src_y0:src_y1, src_x0:src_x1]

            out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 0:3] = asset_crop
            out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 3] = alpha_crop
            out_masks[i, dst_y0:dst_y1, dst_x0:dst_x1] = alpha_crop

        return (out_imgs, out_masks)


NODE_CLASS_MAPPINGS = {"BGRemoveComposeAlpha": BGRemoveComposeAlpha}
NODE_DISPLAY_NAME_MAPPINGS = {"BGRemoveComposeAlpha": "BG Remove + Justify on Alpha"}
