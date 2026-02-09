"""Unified parser for all Minecraft schematic formats.

Reads from a centralized 'houses' directory with subdirectories for each format.
Automatically detects and handles different schematic formats.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Import fixed parsers
try:
    from litematic_parser_fixed import LitematicParser
except ImportError:
    from litematic_parser import LitematicParser

from schem_parser import SchemParser
from schematic_parser import SchematicParser


logger = logging.getLogger(__name__)


class HouseParser:
    """Unified parser for all schematic formats."""

    def __init__(self, houses_dir: str = "houses", cache_dir: str = ".cache"):
        """Initialize the unified parser.

        Args:
            houses_dir: Base directory containing house files in subdirectories.
            cache_dir: Directory to store cache files.
        """
        self.houses_dir = Path(houses_dir)
        self.cache_dir = Path(cache_dir)

        # Initialize format-specific parsers
        self.litematic_parser = LitematicParser(cache_dir=cache_dir)
        self.schem_parser = SchemParser(cache_dir=cache_dir)
        self.schematic_parser = SchematicParser(cache_dir=cache_dir)

        # Directory structure
        self.format_dirs = {
            'litematic': self.houses_dir / 'litematic',
            'schem': self.houses_dir / 'schem',
            'schematic': self.houses_dir / 'schematic'
        }

    def setup_directories(self):
        """Create directory structure if it doesn't exist."""
        self.houses_dir.mkdir(exist_ok=True)
        for format_dir in self.format_dirs.values():
            format_dir.mkdir(exist_ok=True)
        logger.info(f"Directory structure ready at {self.houses_dir}")

    def parse_all(self, use_cache: bool = True) -> Dict[str, Tuple[np.ndarray, Dict[int, str]]]:
        """Parse all house files from all formats.

        Args:
            use_cache: Whether to use cached results if available.

        Returns:
            Dictionary mapping filenames to (voxel_tensor, block_mapping) tuples.
        """
        all_results = {}

        # Parse litematic files
        litematic_dir = self.format_dirs['litematic']
        if litematic_dir.exists() and list(litematic_dir.glob("*.litematic")):
            logger.info("Parsing .litematic files...")
            litematic_results = self.litematic_parser.parse_directory(
                str(litematic_dir), use_cache
            )
            all_results.update(litematic_results)

        # Parse schem files
        schem_dir = self.format_dirs['schem']
        if schem_dir.exists() and list(schem_dir.glob("*.schem")):
            logger.info("Parsing .schem files...")
            schem_results = self.schem_parser.parse_directory(
                str(schem_dir), use_cache
            )
            all_results.update(schem_results)

        # Parse schematic files
        schematic_dir = self.format_dirs['schematic']
        if schematic_dir.exists() and list(schematic_dir.glob("*.schematic")):
            logger.info("Parsing .schematic files...")
            schematic_results = self.schematic_parser.parse_directory(
                str(schematic_dir), use_cache
            )
            all_results.update(schematic_results)

        logger.info(f"Total files parsed: {len(all_results)}")

        if not all_results:
            logger.warning("\n⚠️  No files were successfully parsed!")
            logger.warning("Please check:")
            logger.warning("  1. Files are in correct directories:")
            logger.warning("     - houses/litematic/*.litematic")
            logger.warning("     - houses/schem/*.schem")
            logger.warning("     - houses/schematic/*.schematic")
            logger.warning("  2. Files are not corrupted")
            logger.warning("  3. Files are correct format (not misnamed)")

        return all_results

    def get_unified_block_palette(
        self,
        results: Dict[str, Tuple[np.ndarray, Dict[int, str]]]
    ) -> Dict[str, int]:
        """Create unified block palette across all files.

        Args:
            results: Dictionary from parse_all().

        Returns:
            Dictionary mapping block names to unified block IDs.
        """
        all_blocks = set()

        for _, (_, block_mapping) in results.items():
            all_blocks.update(block_mapping.values())

        # Create unified palette (0 reserved for air)
        unified_palette = {}
        for block_name in sorted(all_blocks):
            if 'air' in block_name.lower() or 'block_0' in block_name.lower():
                unified_palette[block_name] = 0
            else:
                if block_name not in unified_palette:
                    unified_palette[block_name] = len(unified_palette)

        return unified_palette

    def convert_to_unified_ids(
        self,
        voxel_tensor: np.ndarray,
        block_mapping: Dict[int, str],
        unified_palette: Dict[str, int]
    ) -> np.ndarray:
        """Convert voxel tensor to use unified block IDs.

        Args:
            voxel_tensor: Original voxel tensor with file-specific IDs.
            block_mapping: Mapping from file-specific IDs to block names.
            unified_palette: Unified block name to ID mapping.

        Returns:
            Voxel tensor with unified block IDs.
        """
        unified_tensor = np.zeros_like(voxel_tensor)

        for old_id, block_name in block_mapping.items():
            unified_id = unified_palette[block_name]
            unified_tensor[voxel_tensor == old_id] = unified_id

        return unified_tensor

    def convert_all_to_binary(
        self,
        results: Dict[str, Tuple[np.ndarray, Dict[int, str]]]
    ) -> Dict[str, np.ndarray]:
        """Convert all structures to binary (block vs air).

        Args:
            results: Dictionary from parse_all().

        Returns:
            Dictionary mapping filenames to binary voxel tensors (0=air, 1=block).
        """
        binary_results = {}

        for filename, (voxel_tensor, block_mapping) in results.items():
            # Identify air blocks
            air_ids = [
                block_id for block_id, block_name in block_mapping.items()
                if 'air' in block_name.lower() or 'block_0' in block_name.lower()
            ]

            # Create binary tensor
            binary_tensor = np.ones_like(voxel_tensor, dtype=np.int32)
            for air_id in air_ids:
                binary_tensor[voxel_tensor == air_id] = 0

            binary_results[filename] = binary_tensor

        logger.info(f"Converted {len(binary_results)} structures to binary")
        return binary_results


def main():
    """Example usage."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize parser
    parser = HouseParser(houses_dir="houses")

    # Setup directory structure
    parser.setup_directories()

    logger.info("Directory structure:")
    logger.info(f"  houses/litematic  - Place .litematic files here")
    logger.info(f"  houses/schem      - Place .schem files here")
    logger.info(f"  houses/schematic  - Place .schematic files here")

    # Parse all files
    results = parser.parse_all()

    if not results:
        logger.warning("No files found. Please add house files to the directories above.")
        return

    # Show statistics
    logger.info(f"\nParsed {len(results)} house files")

    for filename, (voxel_tensor, block_mapping) in list(results.items())[:3]:
        logger.info(f"\n{filename}:")
        logger.info(f"  Shape: {voxel_tensor.shape}")
        logger.info(f"  Unique blocks: {len(block_mapping)}")

    # Create unified palette
    unified_palette = parser.get_unified_block_palette(results)
    logger.info(f"\nTotal unique blocks: {len(unified_palette)}")

    # Convert to binary
    binary_results = parser.convert_all_to_binary(results)
    logger.info(f"\nBinary conversion complete")


if __name__ == "__main__":
    main()