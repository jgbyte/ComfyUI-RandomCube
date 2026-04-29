import torch
import torch.nn.functional as F

from .bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox, hex_to_rgb


class BGRemoveComposeColor:
    """
    Remove background with BiRefNet/RMBG-2.0, then composite the asset
    centered on a solid-color canvas at user-specified dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODEL_REGISTRY.keys()), {"default": "BiRefNet"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "bg_color": ("STRING", {"default": "#ffffff"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "compose"
    CATEGORY = "image/bg-remove"

    def compose(self, image, model, width, height, bg_color, scale):
        b, _, _, _ = image.shape
        rgb = hex_to_rgb(bg_color)
        bg_rgb = torch.tensor(rgb, dtype=torch.float32) / 255.0

        masks = predict_mask(image, model)

        out_imgs = torch.zeros((b, height, width, 3), dtype=torch.float32)
        out_imgs[:, :, :] = bg_rgb
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

            cy = (height - new_h) // 2
            cx = (width - new_w) // 2

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

            a3 = alpha_crop.unsqueeze(-1)
            canvas_region = out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, :]
            out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, :] = asset_crop * a3 + canvas_region * (1.0 - a3)
            out_masks[i, dst_y0:dst_y1, dst_x0:dst_x1] = alpha_crop

        return (out_imgs, out_masks)


NODE_CLASS_MAPPINGS = {"BGRemoveComposeColor": BGRemoveComposeColor}
NODE_DISPLAY_NAME_MAPPINGS = {"BGRemoveComposeColor": "BG Remove + Center on Color"}
