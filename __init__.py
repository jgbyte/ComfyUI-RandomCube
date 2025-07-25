from .random_cube_grid import NODE_CLASS_MAPPINGS as RANDOM_GRID_NODES, NODE_DISPLAY_NAME_MAPPINGS as RANDOM_GRID_DISPLAY_NAMES
from .offset_image import OffsetImageNode
from .blocks_generator import ColorCubeArrayNode

NODE_CLASS_MAPPINGS = {
    **RANDOM_GRID_NODES,
    "offset_image": OffsetImageNode,
    "blocks_generator": ColorCubeArrayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RANDOM_GRID_DISPLAY_NAMES,
    "offset_image": "Offset Image",
    "blocks_generator": "Block Grid Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
