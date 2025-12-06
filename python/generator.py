"""Minecraft house generator and exporter.

This module builds a simple rectangular house model using NumPy arrays and
exports the result to a JSON structure that can be consumed by the Three.js
viewer in the frontend directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import numpy as np


class BlockType(IntEnum):
    """Enumerates supported block identifiers."""

    AIR = 0
    SOLID = 1
    STAIR = 2


class StairRotation(IntEnum):
    """Represents the orientation of a stair block."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


@dataclass(frozen=True)
class HouseOptions:
    """Holds toggleable generation options."""

    add_windows: bool = True
    add_columns: bool = True
    add_stairs: bool = True
    roof_type: Literal["flat", "gable"] = "flat"


SYMBOLS = {
    BlockType.AIR: ".",
    BlockType.SOLID: "#",
    BlockType.STAIR: "=",
}


def generate_house(
    width: int,
    height: int,
    depth: int,
    options: HouseOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a rectangular house volume.

    Args:
        width: X dimension of the interior footprint.
        height: Wall height in blocks.
        depth: Z dimension of the interior footprint.
        options: Optional overrides for generation toggles.

    Returns:
        Tuple of NumPy arrays representing block ids and stair rotations.

    Raises:
        ValueError: If requested dimensions are invalid.
    """

    if min(width, height, depth) < 2:
        raise ValueError("All dimensions must be at least 2 blocks long.")

    opts = options or HouseOptions()
    grid = np.zeros((width + 2, height + 5, depth + 2), dtype=np.uint8)
    rotation = np.zeros_like(grid)

    origin_x, origin_z = 1, 1

    _build_floor(grid, origin_x, origin_z, width, depth)
    _build_walls(grid, origin_x, origin_z, width, height, depth)
    _carve_doorway(grid, origin_x, origin_z, width)

    if opts.add_windows:
        _carve_windows(grid, origin_x, origin_z, width, height, depth)

    if opts.add_columns:
        _place_corner_columns(grid, origin_x, origin_z, width, height, depth)

    if opts.add_stairs:
        _place_entry_stairs(grid, rotation, origin_x, origin_z, width)

    roof_y = height + 1
    if opts.roof_type == "flat":
        grid[origin_x : origin_x + width, roof_y, origin_z : origin_z + depth] = (
            BlockType.SOLID
        )
    else:
        generate_gable_roof(grid, origin_x, origin_z, roof_y, width, depth)

    return grid, rotation


def _build_floor(grid: np.ndarray, ox: int, oz: int, width: int, depth: int) -> None:
    grid[ox : ox + width, 0, oz : oz + depth] = BlockType.SOLID


def _build_walls(
    grid: np.ndarray,
    ox: int,
    oz: int,
    width: int,
    height: int,
    depth: int,
) -> None:
    grid[ox : ox + width, 1 : height + 1, oz] = BlockType.SOLID
    grid[ox : ox + width, 1 : height + 1, oz + depth - 1] = BlockType.SOLID
    grid[ox, 1 : height + 1, oz : oz + depth] = BlockType.SOLID
    grid[ox + width - 1, 1 : height + 1, oz : oz + depth] = BlockType.SOLID


def _carve_doorway(grid: np.ndarray, ox: int, oz: int, width: int) -> None:
    door_x = ox + width // 2
    grid[door_x, 1:3, oz] = BlockType.AIR


def _carve_windows(
    grid: np.ndarray,
    ox: int,
    oz: int,
    width: int,
    height: int,
    depth: int,
) -> None:
    window_y = min(height, 3)

    for x in range(ox + 1, ox + width - 1, 2):
        grid[x, window_y, oz] = BlockType.AIR
        grid[x, window_y, oz + depth - 1] = BlockType.AIR

    for z in range(oz + 1, oz + depth - 1, 2):
        grid[ox, window_y, z] = BlockType.AIR
        grid[ox + width - 1, window_y, z] = BlockType.AIR


def _place_corner_columns(
    grid: np.ndarray,
    ox: int,
    oz: int,
    width: int,
    height: int,
    depth: int,
) -> None:
    column_positions = (
        (ox - 1, oz - 1),
        (ox + width, oz - 1),
        (ox - 1, oz + depth),
        (ox + width, oz + depth),
    )

    for x, z in column_positions:
        grid[x, 1 : height + 1, z] = BlockType.SOLID


def _place_entry_stairs(
    grid: np.ndarray,
    rotation: np.ndarray,
    ox: int,
    oz: int,
    width: int,
) -> None:
    door_x = ox + width // 2
    step_positions = (door_x, door_x - 1)

    for x in step_positions:
        grid[x, 0, oz - 1] = BlockType.STAIR
        rotation[x, 0, oz - 1] = StairRotation.SOUTH


def generate_gable_roof(
    grid: np.ndarray,
    ox: int,
    oz: int,
    start_y: int,
    width: int,
    depth: int,
) -> None:
    """Create a simple gable roof along the depth axis.

    Args:
        grid: Volume that stores block identifiers.
        ox: Origin X of the interior footprint inside the padded grid.
        oz: Origin Z of the interior footprint inside the padded grid.
        start_y: Vertical layer where the roof begins.
        width: Width of the building footprint.
        depth: Depth of the building footprint.
    """

    half_depth = (depth + 1) // 2

    for layer in range(half_depth):
        y = start_y + layer
        z_start = oz + layer
        z_end = oz + depth - layer
        if z_start >= z_end:
            break
        grid[ox : ox + width, y, z_start:z_end] = BlockType.SOLID


def array_to_json(grid: np.ndarray, rotation: np.ndarray) -> dict:
    """Convert NumPy volumes to a JSON-serializable representation.

    Args:
        grid: Block identifier volume.
        rotation: Stair rotation metadata volume.

    Returns:
        Dictionary ready to be dumped to JSON.
    """

    type_names = {
        BlockType.SOLID: "block",
        BlockType.STAIR: "stair",
    }
    rotation_names = {
        StairRotation.NORTH: "north",
        StairRotation.EAST: "east",
        StairRotation.SOUTH: "south",
        StairRotation.WEST: "west",
    }

    blocks = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                block_id = BlockType(grid[x, y, z])
                if block_id == BlockType.AIR:
                    continue

                block = {
                    "x": int(x - 1),
                    "y": int(y),
                    "z": int(z - 1),
                    "type": type_names[block_id],
                }

                if block_id == BlockType.STAIR:
                    facing = rotation_names[StairRotation(rotation[x, y, z])]
                    block["metadata"] = {
                        "facing": facing,
                        "upsideDown": False,
                    }

                blocks.append(block)

    return {
        "blocks": blocks,
        "dimensions": {
            "width": int(grid.shape[0] - 2),
            "height": int(grid.shape[1] - 5),
            "depth": int(grid.shape[2] - 2),
        },
    }


def visualize_layer(grid: np.ndarray, y_level: int, title: str | None = None) -> None:
    """Print a horizontal slice of the grid for quick inspection.

    Args:
        grid: Block identifier volume.
        y_level: Vertical layer index to display.
        title: Optional caption printed before the slice.
    """

    if title:
        print(title)
    header = "  " + "".join(str(x % 10) for x in range(grid.shape[0]))
    print(header)
    for z in range(grid.shape[2]):
        row = f"{z:2d} "
        for x in range(grid.shape[0]):
            block = BlockType(grid[x, y_level, z])
            row += SYMBOLS[block]
        print(row)


def print_stats(grid: np.ndarray) -> None:
    """Display a short histogram of block usage."""

    unique, counts = np.unique(grid, return_counts=True)
    for block_id, count in zip(unique, counts):
        if block_id == BlockType.AIR:
            continue
        name = BlockType(block_id).name
        print(f"{name:<6}: {count}")


def _default_output_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "house.json"


def export_house_json(data: dict, output_path: Path | None = None) -> Path:
    """Persist the generated JSON to disk."""

    target = output_path or _default_output_path()
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return target


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments provided by the user."""

    parser = argparse.ArgumentParser(description="Generate a Minecraft house.")
    parser.add_argument("--width", type=int, default=8, help="Interior width in blocks")
    parser.add_argument("--height", type=int, default=4, help="Wall height in blocks")
    parser.add_argument("--depth", type=int, default=6, help="Interior depth in blocks")
    parser.add_argument(
        "--roof",
        choices=["flat", "gable"],
        default="flat",
        help="Roof style to apply",
    )
    parser.add_argument("--no-windows", action="store_true", help="Disable window cutouts")
    parser.add_argument("--no-columns", action="store_true", help="Disable corner columns")
    parser.add_argument("--no-stairs", action="store_true", help="Disable entry stairs")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional custom output path for the generated JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for CLI execution."""

    args = parse_args()
    options = HouseOptions(
        add_windows=not args.no_windows,
        add_columns=not args.no_columns,
        add_stairs=not args.no_stairs,
        roof_type=args.roof,
    )

    grid, rotation = generate_house(args.width, args.height, args.depth, options)
    data = array_to_json(grid, rotation)

    target = export_house_json(data, args.output)

    print("House generated:")
    print(f"  Dimensions: {data['dimensions']}")
    print(f"  Blocks: {len(data['blocks'])}")
    print(f"  Output: {target}")
    print()
    visualize_layer(grid, 0, "Floor (Y=0)")
    visualize_layer(grid, args.height // 2, "Mid-wall layer")
    visualize_layer(grid, args.height + 1, "Roof layer")
    print()
    print("Block usage:")
    print_stats(grid)


if __name__ == "__main__":
    main()
