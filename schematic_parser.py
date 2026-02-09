"""Parser for Minecraft .schematic files (MCEdit format)."""

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import nbtlib


logger = logging.getLogger(__name__)


class SchematicParser:
    """Parser for Minecraft .schematic files (old MCEdit format)."""

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
            directory: Path to directory with .schematic files.

        Returns:
            MD5 hash of all file names and modification times.
        """
        schematic_files = sorted(Path(directory).glob("*.schematic"))
        hash_content = ""

        for file in schematic_files:
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

    def _parse_schematic_file(self, filepath: Path) -> Tuple[np.ndarray, Dict[int, str]]:
        """Parse a single .schematic file.

        Args:
            filepath: Path to .schematic file.

        Returns:
            Tuple of (voxel_tensor, block_mapping) where:
                - voxel_tensor: 3D numpy array [X, Y, Z] with block IDs
                - block_mapping: Dict mapping block IDs to block names
        """
        nbt_file = nbtlib.load(filepath)

        # Get dimensions
        width = int(nbt_file['Width'])   # X
        height = int(nbt_file['Height']) # Y
        length = int(nbt_file['Length']) # Z

        # Get blocks and data
        blocks = nbt_file['Blocks']
        data = nbt_file.get('Data', bytes([0] * len(blocks)))

        # Convert to numpy arrays
        blocks_array = np.frombuffer(bytes(blocks), dtype=np.uint8)
        data_array = np.frombuffer(bytes(data), dtype=np.uint8)

        # Combine block ID and data into single value
        combined = (blocks_array.astype(np.int32) << 4) | data_array.astype(np.int32)

        # Create block mapping
        unique_ids = np.unique(combined)
        block_mapping = {}

        # Create mapping from combined IDs to sequential IDs
        for idx, block_id in enumerate(unique_ids):
            block_mapping[idx] = f"minecraft:block_{block_id}"

        # Remap to sequential IDs
        voxel_tensor = np.zeros_like(combined)
        for idx, block_id in enumerate(unique_ids):
            voxel_tensor[combined == block_id] = idx

        # Reshape to 3D array [X, Y, Z]
        # Schematic format is YZX order
        voxel_tensor = voxel_tensor.reshape((height, length, width))
        voxel_tensor = np.transpose(voxel_tensor, (2, 0, 1))  # [X, Y, Z]

        return voxel_tensor, block_mapping

    def parse_directory(
        self,
        directory: str,
        use_cache: bool = True
    ) -> Dict[str, Tuple[np.ndarray, Dict[int, str]]]:
        """Parse all .schematic files in a directory.

        Args:
            directory: Path to directory containing .schematic files.
            use_cache: Whether to use cached results if available.

        Returns:
            Dictionary mapping filenames to (voxel_tensor, block_mapping) tuples.
        """
        directory = Path(directory)
        dir_hash = self._get_directory_hash(directory)
        cache_file = self.cache_dir / f"schematic_cache_{dir_hash}.pkl"

        # Try to load from cache
        if use_cache:
            cached_data = self._load_cache(cache_file)
            if cached_data is not None:
                logger.info(f"Using cached data for {directory}")
                return cached_data

        logger.info(f"Parsing .schematic files from {directory}")

        # Parse all files
        results = {}
        schematic_files = list(directory.glob("*.schematic"))

        for idx, filepath in enumerate(schematic_files, 1):
            logger.info(f"Processing {idx}/{len(schematic_files)}: {filepath.name}")
            try:
                voxel_tensor, block_mapping = self._parse_schematic_file(filepath)
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
        for block_name in sorted(all_blocks):
            if 'air' in block_name.lower() or block_name == 'minecraft:block_0':
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

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python schematic_parser.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    # Initialize parser
    parser = SchematicParser(cache_dir=".schematic_cache")

    # Parse directory
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