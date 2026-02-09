"""Test script for house dataset."""

import logging
import torch
from torch.utils.data import DataLoader
from house_dataset import HouseVoxelDataset


def test_dataset(houses_dir: str = "houses"):
    """Test the house dataset.

    Args:
        houses_dir: Path to houses directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("HOUSE VOXEL DATASET TEST")
    logger.info("=" * 60)

    # Test 1: Binary mode dataset
    logger.info("\n[TEST 1] Binary mode dataset...")
    dataset_binary = HouseVoxelDataset(
        houses_dir=houses_dir,
        binary_mode=True,
        max_size=None  # Auto-detect
    )

    stats = dataset_binary.get_statistics()
    logger.info("\nBinary Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Test 2: Unified ID mode dataset
    logger.info("\n[TEST 2] Unified ID mode dataset...")
    dataset_unified = HouseVoxelDataset(
        houses_dir=houses_dir,
        binary_mode=False,
        max_size=None
    )

    stats = dataset_unified.get_statistics()
    logger.info("\nUnified Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Test 3: Single item access
    logger.info("\n[TEST 3] Single item access...")
    if len(dataset_binary) > 0:
        item = dataset_binary[0]
        logger.info(f"Item keys: {list(item.keys())}")
        logger.info(f"Voxels shape: {item['voxels'].shape}")
        logger.info(f"Voxels dtype: {item['voxels'].dtype}")
        logger.info(f"Voxels min/max: {item['voxels'].min()}/{item['voxels'].max()}")
        logger.info(f"Filename: {item['filename']}")
        logger.info(f"Original shape: {item['original_shape']}")

    # Test 4: DataLoader
    logger.info("\n[TEST 4] DataLoader testing...")
    dataloader = DataLoader(
        dataset_binary,
        batch_size=min(4, len(dataset_binary)),
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(dataloader))
    logger.info(f"Batch size: {len(batch['filename'])}")
    logger.info(f"Batch voxels shape: {batch['voxels'].shape}")
    logger.info(f"Batch voxels dtype: {batch['voxels'].dtype}")
    logger.info(f"Filenames in batch: {batch['filename']}")

    # Test 5: Iteration
    logger.info("\n[TEST 5] Full iteration test...")
    total_voxels = 0
    for i, batch in enumerate(dataloader):
        total_voxels += batch['voxels'].shape[0]
    logger.info(f"Total batches: {i + 1}")
    logger.info(f"Total houses processed: {total_voxels}")

    # Test 6: Custom size
    logger.info("\n[TEST 6] Custom target size...")
    dataset_custom = HouseVoxelDataset(
        houses_dir=houses_dir,
        binary_mode=True,
        max_size=(64, 64, 64)
    )
    item = dataset_custom[0]
    logger.info(f"Custom size voxels shape: {item['voxels'].shape}")
    assert item['voxels'].shape == (1, 64, 64, 64), "Size mismatch!"
    logger.info("Custom size test passed!")

    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    houses_dir = sys.argv[1] if len(sys.argv) > 1 else "houses"
    test_dataset(houses_dir)