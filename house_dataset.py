"""PyTorch dataset for Minecraft house voxels."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

from unified_parser import HouseParser

logger = logging.getLogger(__name__)


class HouseVoxelDataset(Dataset):
    """PyTorch dataset for Minecraft house voxels."""

    def __init__(
            self,
            houses_dir: str = "houses",
            cache_dir: str = ".cache",
            binary_mode: bool = True,
            use_cache: bool = True,
            max_size: Optional[Tuple[int, int, int]] = None,
            padding_mode: str = "constant"
    ):
        """Initialize the dataset.

        Args:
            houses_dir: Directory containing house files.
            cache_dir: Directory for parser cache.
            binary_mode: If True, convert to binary (0=air, 1=block).
                        If False, use unified block IDs.
            use_cache: Whether to use parser cache.
            max_size: Optional (X, Y, Z) to pad/crop all houses to.
                     If None, uses maximum dimensions from dataset.
            padding_mode: Padding mode for torch.nn.functional.pad.
        """
        self.binary_mode = binary_mode
        self.max_size = max_size
        self.padding_mode = padding_mode

        # Parse all houses
        logger.info("Initializing dataset...")
        self.parser = HouseParser(houses_dir=houses_dir, cache_dir=cache_dir)
        self.parser.setup_directories()

        self.raw_data = self.parser.parse_all(use_cache=use_cache)

        if not self.raw_data:
            raise ValueError(f"No house files found in {houses_dir}")

        # Convert data based on mode
        if binary_mode:
            logger.info("Converting to binary mode...")
            self.data = self.parser.convert_all_to_binary(self.raw_data)
        else:
            logger.info("Converting to unified IDs...")
            self.unified_palette = self.parser.get_unified_block_palette(self.raw_data)
            self.data = {}
            for filename, (voxel_tensor, block_mapping) in self.raw_data.items():
                unified_tensor = self.parser.convert_to_unified_ids(
                    voxel_tensor, block_mapping, self.unified_palette
                )
                self.data[filename] = unified_tensor

        # Store filenames for indexing
        self.filenames = list(self.data.keys())

        # Calculate max dimensions if not provided
        if self.max_size is None:
            max_x = max(tensor.shape[0] for tensor in self.data.values())
            max_y = max(tensor.shape[1] for tensor in self.data.values())
            max_z = max(tensor.shape[2] for tensor in self.data.values())
            self.max_size = (max_x, max_y, max_z)
            logger.info(f"Auto-detected max size: {self.max_size}")

        logger.info(f"Dataset initialized with {len(self)} houses")
        logger.info(f"Target size: {self.max_size}")

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of houses in dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Index of house to retrieve.

        Returns:
            Dictionary containing:
                - 'voxels': Padded/cropped voxel tensor [1, X, Y, Z]
                - 'filename': Name of the house file
                - 'original_shape': Original shape before padding/cropping
        """
        filename = self.filenames[idx]
        voxel_tensor = self.data[filename]

        # Store original shape
        original_shape = voxel_tensor.shape

        # Pad or crop to max_size
        voxel_tensor = self._resize_voxels(voxel_tensor, self.max_size)

        # Convert to torch tensor and add channel dimension
        voxel_tensor = torch.from_numpy(voxel_tensor).float()
        voxel_tensor = voxel_tensor.unsqueeze(0)  # [1, X, Y, Z]

        return {
            'voxels': voxel_tensor,
            'filename': filename,
            'original_shape': original_shape
        }

    def _resize_voxels(
            self,
            voxels: np.ndarray,
            target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize voxels to target size by padding or cropping.

        Args:
            voxels: Input voxel array [X, Y, Z].
            target_size: Target (X, Y, Z) dimensions.

        Returns:
            Resized voxel array.
        """
        x, y, z = voxels.shape
        tx, ty, tz = target_size

        # Crop if larger
        if x > tx:
            voxels = voxels[:tx, :, :]
        if y > ty:
            voxels = voxels[:, :ty, :]
        if z > tz:
            voxels = voxels[:, :, :tz]

        # Pad if smaller
        x, y, z = voxels.shape
        if x < tx or y < ty or z < tz:
            pad_x = tx - x
            pad_y = ty - y
            pad_z = tz - z

            # Convert to tensor for padding
            voxels_tensor = torch.from_numpy(voxels)

            # Pad in reverse order: (z_left, z_right, y_left, y_right, x_left, x_right)
            padding = (0, pad_z, 0, pad_y, 0, pad_x)
            voxels_tensor = torch.nn.functional.pad(
                voxels_tensor,
                padding,
                mode=self.padding_mode,
                value=0
            )

            voxels = voxels_tensor.numpy()

        return voxels

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            'num_houses': len(self),
            'target_size': self.max_size,
            'binary_mode': self.binary_mode,
        }

        if not self.binary_mode:
            stats['num_unique_blocks'] = len(self.unified_palette)

        # Calculate size distribution
        shapes = [self.data[fn].shape for fn in self.filenames]
        stats['min_shape'] = tuple(np.min(shapes, axis=0))
        stats['max_shape'] = tuple(np.max(shapes, axis=0))
        stats['mean_shape'] = tuple(np.mean(shapes, axis=0).astype(int))

        # Calculate density (for binary mode)
        if self.binary_mode:
            densities = [
                (self.data[fn] == 1).sum() / self.data[fn].size
                for fn in self.filenames
            ]
            stats['mean_density'] = np.mean(densities)
            stats['min_density'] = np.min(densities)
            stats['max_density'] = np.max(densities)

        return stats


def main():
    """Example usage."""
    import logging
    from torch.utils.data import DataLoader

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create dataset
    dataset = HouseVoxelDataset(
        houses_dir="houses",
        binary_mode=True,
        max_size=None  # Auto-detect
    )

    # Show statistics
    stats = dataset.get_statistics()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    # Test batch
    logger.info("\nTesting batch loading...")
    batch = next(iter(dataloader))
    logger.info(f"Batch voxels shape: {batch['voxels'].shape}")
    logger.info(f"Batch filenames: {batch['filename']}")

    # Test individual item
    logger.info("\nTesting individual item...")
    item = dataset[0]
    logger.info(f"Item voxels shape: {item['voxels'].shape}")
    logger.info(f"Item filename: {item['filename']}")
    logger.info(f"Original shape: {item['original_shape']}")


if __name__ == "__main__":
    main()