"""Test script for nbt parser."""

from nbt_parser import NBTParser
import sys
import numpy as np


def test_nbt_parser(directory: str):
    """Test the nbt parser on a directory.

    Args:
        directory: Path to directory with .nbt files.
    """
    print("=" * 60)
    print("NBT PARSER TEST")
    print("=" * 60)

    # Initialize parser
    parser = NBTParser(cache_dir=".nbt_cache")

    # First run - parses files
    print("\n[TEST 1] First run - parsing files...")
    results = parser.parse_directory(directory, use_cache=False)

    if not results:
        print("No .nbt files found!")
        return

    # Second run - should use cache
    print("\n[TEST 2] Second run - testing cache...")
    results = parser.parse_directory(directory, use_cache=True)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total files parsed: {len(results)}")

    # Analyze each file
    for filename, (voxel_tensor, block_mapping) in results.items():
        print(f"\n{filename}:")
        print(f"  Shape (X,Y,Z): {voxel_tensor.shape}")
        print(f"  Total blocks: {voxel_tensor.size}")
        print(f"  Unique block types: {len(block_mapping)}")
        print(f"  Min block ID: {voxel_tensor.min()}")
        print(f"  Max block ID: {voxel_tensor.max()}")

        # Count non-zero blocks (placed blocks)
        non_zero = np.count_nonzero(voxel_tensor)
        print(f"  Placed blocks: {non_zero} ({100 * non_zero / voxel_tensor.size:.1f}%)")

        # Count air vs solid blocks
        air_blocks = sum(1 for name in block_mapping.values() if 'air' in name.lower())
        solid_blocks = len(block_mapping) - air_blocks
        print(f"  Air types: {air_blocks}, Solid types: {solid_blocks}")

        # Show first 5 block types
        print(f"  Sample blocks: {list(block_mapping.values())[:5]}")

    # Create unified palette
    print("\n" + "=" * 60)
    print("UNIFIED PALETTE")
    print("=" * 60)
    unified_palette = parser.get_unified_block_palette(results)
    print(f"Total unique blocks across all files: {len(unified_palette)}")

    # Show first 10 blocks
    print("\nFirst 10 block types:")
    for block_name, block_id in list(unified_palette.items())[:10]:
        print(f"  {block_id}: {block_name}")

    # Convert one example to unified IDs
    if results:
        filename, (voxel_tensor, block_mapping) = next(iter(results.items()))
        unified_tensor = parser.convert_to_unified_ids(
            voxel_tensor,
            block_mapping,
            unified_palette
        )
        print(f"\n[TEST 3] Conversion to unified IDs:")
        print(f"  Original shape: {voxel_tensor.shape}")
        print(f"  Unified shape: {unified_tensor.shape}")
        print(f"  Unified min ID: {unified_tensor.min()}")
        print(f"  Unified max ID: {unified_tensor.max()}")

        # Verify conversion
        print(f"  Shape preserved: {voxel_tensor.shape == unified_tensor.shape}")
        print(
            f"  Data converted: {not np.array_equal(voxel_tensor, unified_tensor) or len(block_mapping) == len(unified_palette)}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_nbt_parser.py <directory_with_nbt_files>")
        sys.exit(1)

    test_nbt_parser(sys.argv[1])