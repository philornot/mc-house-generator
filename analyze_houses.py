"""Analyze all generated houses in a directory.

Quick summary of generated houses - count, sizes, densities, etc.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from tabulate import tabulate

logger = logging.getLogger(__name__)


def analyze_houses(directory: str, verbose: bool = False):
    """Analyze all .npy house files in directory.

    Args:
        directory: Path to directory with .npy files.
        verbose: If True, show details for each house.
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory {directory} not found!")
        return

    # Find all .npy files
    npy_files = sorted(list(directory.glob('*.npy')))

    if not npy_files:
        logger.warning(f"No .npy files found in {directory}")
        return

    logger.info(f"Found {len(npy_files)} house file(s)")
    logger.info("=" * 70)

    # Analyze each house
    house_stats = []
    total_blocks = 0

    for i, npy_file in enumerate(npy_files, 1):
        voxels = np.load(npy_file)

        # Handle both single houses and batches
        if voxels.ndim == 4:
            # Multiple houses in one file
            num_houses_in_file = voxels.shape[0]
            logger.info(f"\n{npy_file.name}: {num_houses_in_file} houses")

            for j in range(num_houses_in_file):
                house = voxels[j]
                blocks = (house > 0.5).sum()
                density = blocks / house.size

                house_stats.append({
                    'file': f"{npy_file.name}[{j}]",
                    'shape': house.shape,
                    'blocks': int(blocks),
                    'density': density
                })

                total_blocks += blocks

                if verbose:
                    logger.info(f"  [{j}] Shape: {house.shape}, "
                                f"Blocks: {int(blocks)}, "
                                f"Density: {density:.2%}")

        else:
            # Single house
            blocks = (voxels > 0.5).sum()
            density = blocks / voxels.size

            house_stats.append({
                'file': npy_file.name,
                'shape': voxels.shape,
                'blocks': int(blocks),
                'density': density
            })

            total_blocks += blocks

            if verbose:
                logger.info(f"\n{npy_file.name}:")
                logger.info(f"  Shape: {voxels.shape}")
                logger.info(f"  Blocks: {int(blocks)}")
                logger.info(f"  Density: {density:.2%}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)

    logger.info(f"\nTotal houses: {len(house_stats)}")
    logger.info(f"Total blocks: {int(total_blocks):,}")

    # Density statistics
    densities = [h['density'] for h in house_stats]
    logger.info(f"\nDensity statistics:")
    logger.info(f"  Mean: {np.mean(densities):.2%}")
    logger.info(f"  Min:  {np.min(densities):.2%}")
    logger.info(f"  Max:  {np.max(densities):.2%}")
    logger.info(f"  Std:  {np.std(densities):.2%}")

    # Block count statistics
    block_counts = [h['blocks'] for h in house_stats]
    logger.info(f"\nBlock count statistics:")
    logger.info(f"  Mean: {np.mean(block_counts):.0f}")
    logger.info(f"  Min:  {np.min(block_counts):.0f}")
    logger.info(f"  Max:  {np.max(block_counts):.0f}")
    logger.info(f"  Std:  {np.std(block_counts):.0f}")

    # Size distribution
    shapes = [h['shape'] for h in house_stats]
    unique_shapes = set(shapes)
    logger.info(f"\nUnique shapes: {len(unique_shapes)}")
    if len(unique_shapes) <= 5:
        for shape in unique_shapes:
            count = shapes.count(shape)
            logger.info(f"  {shape}: {count} house(s)")

    # Quality check
    logger.info("\n" + "=" * 70)
    logger.info("QUALITY CHECK")
    logger.info("=" * 70)

    # Check for empty houses
    empty_houses = [h for h in house_stats if h['blocks'] == 0]
    if empty_houses:
        logger.warning(f"\n⚠️  Found {len(empty_houses)} EMPTY houses!")
        for h in empty_houses[:5]:
            logger.warning(f"  - {h['file']}")
    else:
        logger.info("\n✅ No empty houses")

    # Check for full houses
    full_houses = [h for h in house_stats if h['density'] > 0.95]
    if full_houses:
        logger.warning(f"\n⚠️  Found {len(full_houses)} almost FULL houses (>95% density)!")
        for h in full_houses[:5]:
            logger.warning(f"  - {h['file']}: {h['density']:.2%}")
    else:
        logger.info("✅ No overly dense houses")

    # Check for very sparse houses
    sparse_houses = [h for h in house_stats if 0 < h['density'] < 0.05]
    if sparse_houses:
        logger.warning(f"\n⚠️  Found {len(sparse_houses)} very SPARSE houses (<5% density)!")
        for h in sparse_houses[:5]:
            logger.warning(f"  - {h['file']}: {h['density']:.2%}")
    else:
        logger.info("✅ No overly sparse houses")

    # Reasonable density range check
    good_houses = [h for h in house_stats if 0.1 <= h['density'] <= 0.6]
    logger.info(f"\n✅ {len(good_houses)}/{len(house_stats)} houses have "
                f"reasonable density (10-60%)")

    # Create detailed table if requested
    if verbose and len(house_stats) <= 50:
        logger.info("\n" + "=" * 70)
        logger.info("DETAILED TABLE")
        logger.info("=" * 70 + "\n")

        table_data = []
        for i, h in enumerate(house_stats, 1):
            table_data.append([
                i,
                h['file'],
                f"{h['shape']}",
                f"{h['blocks']:,}",
                f"{h['density']:.1%}"
            ])

        print(tabulate(
            table_data,
            headers=['#', 'File', 'Shape', 'Blocks', 'Density'],
            tablefmt='simple'
        ))


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze generated houses in a directory'
    )
    parser.add_argument('directory', type=str,
                        help='Directory with .npy house files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed information for each house')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    analyze_houses(args.directory, verbose=args.verbose)


if __name__ == "__main__":
    main()