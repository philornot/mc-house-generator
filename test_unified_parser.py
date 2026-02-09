"""Test script for unified parser."""

import logging
from unified_parser import HouseParser
import sys


def test_unified_parser(houses_dir: str = "houses"):
    """Test the unified parser.

    Args:
        houses_dir: Path to houses directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("UNIFIED HOUSE PARSER TEST")
    logger.info("=" * 60)

    # Initialize parser
    parser = HouseParser(houses_dir=houses_dir)

    # Setup directories
    parser.setup_directories()

    # Test 1: Parse all files (no cache)
    logger.info("\n[TEST 1] Parsing all files (no cache)...")
    results = parser.parse_all(use_cache=False)

    if not results:
        logger.warning("No files found!")
        logger.info("\nPlace your house files in:")
        logger.info(f"  {houses_dir}/litematic/  - for .litematic files")
        logger.info(f"  {houses_dir}/schem/      - for .schem files")
        logger.info(f"  {houses_dir}/schematic/  - for .schematic files")
        return

    # Test 2: Parse with cache
    logger.info("\n[TEST 2] Parsing with cache...")
    results = parser.parse_all(use_cache=True)

    # Show results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total houses: {len(results)}")

    for filename, (voxel_tensor, block_mapping) in results.items():
        logger.info(f"\n{filename}:")
        logger.info(f"  Shape (X,Y,Z): {voxel_tensor.shape}")
        logger.info(f"  Total voxels: {voxel_tensor.size}")
        logger.info(f"  Unique blocks: {len(block_mapping)}")
        logger.info(f"  Block range: {voxel_tensor.min()} - {voxel_tensor.max()}")

    # Test 3: Unified palette
    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED PALETTE")
    logger.info("=" * 60)
    unified_palette = parser.get_unified_block_palette(results)
    logger.info(f"Total unique blocks: {len(unified_palette)}")

    # Show first 10 blocks
    logger.info("\nFirst 10 blocks:")
    for block_name, block_id in list(unified_palette.items())[:10]:
        logger.info(f"  {block_id}: {block_name}")

    # Test 4: Convert to unified IDs
    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED ID CONVERSION")
    logger.info("=" * 60)

    if results:
        filename, (voxel_tensor, block_mapping) = next(iter(results.items()))
        unified_tensor = parser.convert_to_unified_ids(
            voxel_tensor, block_mapping, unified_palette
        )
        logger.info(f"Example: {filename}")
        logger.info(f"  Original range: {voxel_tensor.min()} - {voxel_tensor.max()}")
        logger.info(f"  Unified range: {unified_tensor.min()} - {unified_tensor.max()}")

    # Test 5: Binary conversion
    logger.info("\n" + "=" * 60)
    logger.info("BINARY CONVERSION (for initial ML training)")
    logger.info("=" * 60)

    binary_results = parser.convert_all_to_binary(results)

    if binary_results:
        filename, binary_tensor = next(iter(binary_results.items()))
        logger.info(f"Example: {filename}")
        logger.info(f"  Shape: {binary_tensor.shape}")
        logger.info(f"  Air voxels: {(binary_tensor == 0).sum()}")
        logger.info(f"  Block voxels: {(binary_tensor == 1).sum()}")
        density = (binary_tensor == 1).sum() / binary_tensor.size
        logger.info(f"  Density: {density:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    houses_dir = sys.argv[1] if len(sys.argv) > 1 else "houses"
    test_unified_parser(houses_dir)