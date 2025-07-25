from PIL import Image, ImageDraw
import torch
import re

class blocks_generator.py:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "array_width": ("INT", {"default": 10, "min": 1, "max": 100}),
                "array_height": ("INT", {"default": 10, "min": 1, "max": 100}),
                "resolution": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 8}),
                "bg_color": ("STRING", {"default": "#000000"}),
                "color1": ("STRING", {"default": "#ff0000"}),
                "coords1": ("STRING", {"multiline": True, "default": ""}),
                "color2": ("STRING", {"default": "#00ff00"}),
                "coords2": ("STRING", {"multiline": True, "default": ""}),
                "color3": ("STRING", {"default": "#0000ff"}),
                "coords3": ("STRING", {"multiline": True, "default": ""}),
                "color4": ("STRING", {"default": "#ffff00"}),
                "coords4": ("STRING", {"multiline": True, "default": ""}),
                "color5": ("STRING", {"default": "#ff00ff"}),
                "coords5": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_cubes"
    CATEGORY = "image/generation"

    def draw_cubes(self, array_width, array_height, resolution,
                   bg_color, color1, coords1,
                   color2, coords2,
                   color3, coords3,
                   color4, coords4,
                   color5, coords5):

        # Determine final image size
        cell_size = resolution // max(array_width, array_height)
        img_width = array_width * cell_size
        img_height = array_height * cell_size
        image = Image.new("RGB", (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(image)

        # Helper to parse coordinates
        def parse_coords(text):
            matches = re.findall(r"(\d+),\s*(\d+)", text)
            return [(int(x), int(y)) for x, y in matches]

        # Draw squares
        for color, coords in zip(
            [color1, color2, color3, color4, color5],
            [coords1, coords2, coords3, coords4, coords5]
        ):
            for x, y in parse_coords(coords):
                if 0 <= x < array_width and 0 <= y < array_height:
                    draw.rectangle(
                        [x * cell_size, y * cell_size,
                         (x + 1) * cell_size - 1, (y + 1) * cell_size - 1],
                        fill=color
                    )

        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return (image_tensor,)
