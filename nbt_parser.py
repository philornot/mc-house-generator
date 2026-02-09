"""Parser for Minecraft .nbt files (Structure Block format)."""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import nbtlib


class NBTParser:
    """Parser for Minecraft .nbt structure files (Structure Block format)."""

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
            directory: Path to directory with .nbt files.

        Returns:
            MD5 hash of all file names and modification times.
        """
        nbt_files = sorted(Path(directory).glob("*.nbt"))
        hash_content = ""

        for file in nbt_files:
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

    def _parse_nbt_file(self, filepath: Path) -> Tuple[np.ndarray, Dict[int, str]]:
        """Parse a single .nbt structure file.

        Args:
            filepath: Path to .nbt file.

        Returns:
            Tuple of (voxel_tensor, block_mapping) where:
                - voxel_tensor: 3D numpy array [X, Y, Z] with block IDs
                - block_mapping: Dict mapping block IDs to block names
        """
        nbt_file = nbtlib.load(filepath)

        # Get dimensions
        size = nbt_file['size']
        x_size = int(size[0])
        y_size = int(size[1])
        z_size = int(size[2])

        # Get palette
        palette_nbt = nbt_file['palette']
        block_mapping = {}

        for idx, block_state in enumerate(palette_nbt):
            block_name = str(block_state['Name'])
            block_mapping[idx] = block_name

        # Get blocks
        blocks = nbt_file['blocks']

        # Create empty voxel tensor
        voxel_tensor = np.zeros((x_size, y_size, z_size), dtype=np.int32)

        # Fill tensor with blocks
        for block in blocks:
            pos = block['pos']
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            state = int(block['state'])

            if 0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size:
                voxel_tensor[x, y, z] = state

        return voxel_tensor, block_mapping

    def parse_directory(
            self,
            directory: str,
            use_cache: bool = True
    ) -> Dict[str, Tuple[np.ndarray, Dict[int, str]]]:
        """Parse all .nbt files in a directory.

        Args:
            directory: Path to directory containing .nbt files.
            use_cache: Whether to use cached results if available.

        Returns:
            Dictionary mapping filenames to (voxel_tensor, block_mapping) tuples.
        """
        directory = Path(directory)
        dir_hash = self._get_directory_hash(directory)
        cache_file = self.cache_dir / f"nbt_cache_{dir_hash}.pkl"

        # Try to load from cache
        if use_cache:
            cached_data = self._load_cache(cache_file)
            if cached_data is not None:
                print(f"Using cached data for {directory}")
                return cached_data

        print(f"Parsing .nbt files from {directory}")

        # Parse all files
        results = {}
        nbt_files = list(directory.glob("*.nbt"))

        for idx, filepath in enumerate(nbt_files, 1):
            print(f"Processing {idx}/{len(nbt_files)}: {filepath.name}")
            try:
                voxel_tensor, block_mapping = self._parse_nbt_file(filepath)
                results[filepath.name] = (voxel_tensor, block_mapping)
            except Exception as e:
                print(f"Error parsing {filepath.name}: {e}")

        # Save to cache
        self._save_cache(cache_file, results)
        print(f"Cached results to {cache_file}")

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
        for block_name in sorted(all_blocks):
            if 'air' in block_name.lower():
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


def main():
    """Example usage of the parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nbt_parser.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    # Initialize parser
    parser = NBTParser(cache_dir=".nbt_cache")

    # Parse directory
    results = parser.parse_directory(directory)

    print(f"\nParsed {len(results)} files")

    # Show example data
    if results:
        filename, (voxel_tensor, block_mapping) = next(iter(results.items()))
        print(f"\nExample: {filename}")
        print(f"Shape: {voxel_tensor.shape}")
        print(f"Unique blocks: {len(block_mapping)}")
        print(f"Block types: {list(block_mapping.values())[:5]}...")

        # Create unified palette
        unified_palette = parser.get_unified_block_palette(results)
        print(f"\nTotal unique blocks across all files: {len(unified_palette)}")


if __name__ == "__main__":
    main()