"""Litematic file parser for Minecraft schematics.

This parser converts .litematic files into voxel tensors for machine learning.
Includes caching mechanism to avoid reprocessing unchanged directories.
"""

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from litemapy import Schematic, Region


logger = logging.getLogger(__name__)


class LitematicParser:
    """Parser for Minecraft .litematic schematic files."""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the parser.

        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_directory_hash(self, directory: str) -> str:
        """Calculate hash of directory contents.

        Args:
            directory: Path to directory with .litematic files.

        Returns:
            MD5 hash of all file names and modification times.
        """
        litematic_files = sorted(Path(directory).glob("*.litematic"))
        hash_content = ""

        for file in litematic_files:
            stat = file.stat()
            hash_content += f"{file.name}:{stat.st_mtime}:{stat.st_size}\n"

        return hashlib.md5(hash_content.encode()).hexdigest()

    def _load_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load cached data.

        Args:
            cache_file: Path to cache file.

        Returns:
            Cached data dictionary or None if cache doesn't exist.
        """
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save data to cache.

        Args:
            cache_file: Path to cache file.
            data: Data dictionary to cache.
        """
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _parse_litematic_file(self, filepath: Path) -> Tuple[np.ndarray, Dict[int, str]]:
        """Parse a single .litematic file.

        Args:
            filepath: Path to .litematic file.

        Returns:
            Tuple of (voxel_tensor, block_mapping) where:
                - voxel_tensor: 3D numpy array [X, Y, Z] with block IDs
                - block_mapping: Dict mapping block IDs to block names
        """
        # Load schematic using litemapy
        schem = Schematic.load(str(filepath))

        # Get first region (most schematics have one region)
        region = list(schem.regions.values())[0]

        # Get dimensions
        x_size, y_size, z_size = region.width, region.height, region.length

        # Create voxel tensor
        voxel_tensor = np.zeros((x_size, y_size, z_size), dtype=np.int32)

        # Create block palette
        block_to_id = {}
        block_mapping = {}
        current_id = 0

        # Iterate through all blocks in region
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    block = region.getblock(x, y, z)

                    # Get block name (with properties if present)
                    if hasattr(block, 'blockid'):
                        block_name = block.blockid
                    else:
                        block_name = str(block)

                    # Add to palette if new
                    if block_name not in block_to_id:
                        block_to_id[block_name] = current_id
                        block_mapping[current_id] = block_name
                        current_id += 1

                    # Set voxel value
                    voxel_tensor[x, y, z] = block_to_id[block_name]

        return voxel_tensor, block_mapping

    def parse_directory(
        self,
        directory: str,
        use_cache: bool = True
    ) -> Dict[str, Tuple[np.ndarray, Dict[int, str]]]:
        """Parse all .litematic files in a directory.

        Args:
            directory: Path to directory containing .litematic files.
            use_cache: Whether to use cached results if available.

        Returns:
            Dictionary mapping filenames to (voxel_tensor, block_mapping) tuples.
        """
        directory = Path(directory)
        dir_hash = self._get_directory_hash(directory)
        cache_file = self.cache_dir / f"cache_{dir_hash}.pkl"

        # Try to load from cache
        if use_cache:
            cached_data = self._load_cache(cache_file)
            if cached_data is not None:
                logger.info(f"Using cached data for {directory}")
                return cached_data

        logger.info(f"Parsing .litematic files from {directory}")

        # Parse all files
        results = {}
        litematic_files = list(directory.glob("*.litematic"))

        for idx, filepath in enumerate(litematic_files, 1):
            logger.info(f"Processing {idx}/{len(litematic_files)}: {filepath.name}")
            try:
                voxel_tensor, block_mapping = self._parse_litematic_file(filepath)
                results[filepath.name] = (voxel_tensor, block_mapping)
            except Exception as e:
                logger.error(f"Error parsing {filepath.name}: {e}")

        # Save to cache
        self._save_cache(cache_file, results)
        logger.info(f"Cached results to {cache_file}")

        return results

    def get_unified_block_palette(
        self,
        results: Dict[str, Tuple[np.ndarray, Dict[int, str]]]
    ) -> Dict[str, int]:
        """Create unified block palette across all files.

        Args:
            results: Dictionary from parse_directory().

        Returns:
            Dictionary mapping block names to unified block IDs.
        """
        all_blocks = set()

        for _, (_, block_mapping) in results.items():
            all_blocks.update(block_mapping.values())

        # Create unified palette (0 reserved for air)
        unified_palette = {}
        for idx, block_name in enumerate(sorted(all_blocks)):
            if 'air' in block_name.lower():
                unified_palette[block_name] = 0
            else:
                # Assign IDs starting from 1
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


def main():
    """Example usage of the parser."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python litematic_parser.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    # Initialize parser
    parser = LitematicParser(cache_dir=".litematic_cache")

    # Parse directory (uses cache automatically)
    results = parser.parse_directory(directory)

    logger.info(f"Parsed {len(results)} files")

    # Show example data
    if results:
        filename, (voxel_tensor, block_mapping) = next(iter(results.items()))
        logger.info(f"Example: {filename}")
        logger.info(f"Shape: {voxel_tensor.shape}")
        logger.info(f"Unique blocks: {len(block_mapping)}")
        logger.info(f"Block types: {list(block_mapping.values())[:5]}...")

        # Create unified palette
        unified_palette = parser.get_unified_block_palette(results)
        logger.info(f"Total unique blocks across all files: {len(unified_palette)}")


if __name__ == "__main__":
    main()