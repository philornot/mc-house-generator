"""Diagnose why your VAE generates weird houses.

This script checks your training data and model to find issues.
"""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def check_training_data(houses_dir: str = "houses"):
    """Check training data quality.

    Args:
        houses_dir: Path to houses directory.
    """
    from unified_parser import HouseParser

    logger.info("=" * 70)
    logger.info("TRAINING DATA DIAGNOSIS")
    logger.info("=" * 70)

    parser = HouseParser(houses_dir=houses_dir)
    parser.setup_directories()

    # Parse all houses
    results = parser.parse_all(use_cache=True)

    if not results:
        logger.error("\n❌ NO TRAINING DATA FOUND!")
        logger.error("\nYou need to add house files to:")
        logger.error(f"  {houses_dir}/litematic/")
        logger.error(f"  {houses_dir}/schem/")
        logger.error(f"  {houses_dir}/schematic/")
        return False

    # Convert to binary
    binary_results = parser.convert_all_to_binary(results)

    # Analyze
    num_houses = len(binary_results)
    densities = []
    block_counts = []
    shapes = []

    for filename, voxels in binary_results.items():
        density = (voxels == 1).sum() / voxels.size
        blocks = (voxels == 1).sum()

        densities.append(density)
        block_counts.append(blocks)
        shapes.append(voxels.shape)

    # Report
    logger.info(f"\n📊 TRAINING DATA SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total houses: {num_houses}")
    logger.info(f"\nDensity statistics:")
    logger.info(f"  Mean:   {np.mean(densities):.2%}")
    logger.info(f"  Median: {np.median(densities):.2%}")
    logger.info(f"  Min:    {np.min(densities):.2%}")
    logger.info(f"  Max:    {np.max(densities):.2%}")

    logger.info(f"\nBlock count statistics:")
    logger.info(f"  Mean:   {np.mean(block_counts):.0f}")
    logger.info(f"  Median: {np.median(block_counts):.0f}")
    logger.info(f"  Min:    {np.min(block_counts):.0f}")
    logger.info(f"  Max:    {np.max(block_counts):.0f}")

    # Warnings
    logger.info(f"\n⚠️  WARNINGS")
    logger.info(f"{'=' * 70}")

    issues_found = False

    if num_houses < 10:
        logger.warning(f"❌ Only {num_houses} houses - need at least 20 for decent results!")
        logger.warning("   With <10 houses, model will just memorize and generate blobs.")
        issues_found = True
    elif num_houses < 20:
        logger.warning(f"⚠️  {num_houses} houses - barely enough. 30+ recommended.")
        issues_found = True
    else:
        logger.info(f"✅ {num_houses} houses - good amount!")

    if np.mean(densities) > 0.3:
        logger.warning(f"❌ High average density ({np.mean(densities):.2%})!")
        logger.warning("   Model will learn to generate very dense structures.")
        logger.warning("   Are your training houses mostly solid blocks?")
        issues_found = True
    elif np.mean(densities) < 0.05:
        logger.warning(f"❌ Very low average density ({np.mean(densities):.2%})!")
        logger.warning("   Model will generate mostly empty space.")
        issues_found = True
    else:
        logger.info(f"✅ Good average density ({np.mean(densities):.2%})")

    if np.std(densities) < 0.02:
        logger.warning(f"❌ Very similar densities (std={np.std(densities):.3f})!")
        logger.warning("   All houses look too similar. Need more diversity.")
        issues_found = True
    else:
        logger.info(f"✅ Good density variation (std={np.std(densities):.3f})")

    if not issues_found:
        logger.info("✅ No issues found with training data!")

    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS")
    logger.info(f"{'=' * 70}")

    if num_houses < 20:
        logger.info("1. COLLECT MORE HOUSES!")
        logger.info("   - Build 10-20 more diverse houses in Minecraft")
        logger.info("   - Use Litematica or WorldEdit to save them")
        logger.info("   - Try different styles: modern, medieval, cottage, etc.")

    if np.mean(densities) > 0.2:
        logger.info("2. Use higher threshold when generating:")
        logger.info("   python generate_houses.py --threshold 0.6")
    elif np.mean(densities) < 0.1:
        logger.info("2. Use lower threshold when generating:")
        logger.info("   python generate_houses.py --threshold 0.4")

    logger.info("\n3. After fixing vae_model.py bug, retrain:")
    logger.info("   rm -rf checkpoints/  # Delete old checkpoints")
    logger.info("   python train_vae.py --epochs 100 --kl_weight 0.002")

    return True


def check_generated_samples(generated_dir: str = "generated_houses"):
    """Check quality of generated samples.

    Args:
        generated_dir: Path to directory with generated .npy files.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATED SAMPLES CHECK")
    logger.info(f"{'=' * 70}")

    generated_path = Path(generated_dir)
    if not generated_path.exists():
        logger.warning(f"⚠️  No generated samples found at {generated_dir}")
        logger.info("   Generate some first:")
        logger.info("   python generate_houses.py --checkpoint checkpoints/best_model.pth --num_samples 10")
        return

    # Load samples
    npy_files = list(generated_path.glob('*.npy'))
    if not npy_files:
        logger.warning(f"⚠️  No .npy files found in {generated_dir}")
        return

    logger.info(f"Found {len(npy_files)} generated house file(s)")

    densities = []
    block_counts = []

    for npy_file in npy_files[:10]:  # Check first 10
        voxels = np.load(npy_file)

        # Handle batch dimension
        if voxels.ndim == 4:
            voxels = voxels[0]

        density = (voxels > 0.5).sum() / voxels.size
        blocks = (voxels > 0.5).sum()

        densities.append(density)
        block_counts.append(blocks)

    logger.info(f"\nGenerated samples statistics:")
    logger.info(f"  Mean density: {np.mean(densities):.2%}")
    logger.info(f"  Mean blocks:  {np.mean(block_counts):.0f}")

    # Check for issues
    if np.mean(densities) > 0.5:
        logger.error("❌ Generated houses are >50% dense!")
        logger.error("   This is the 'dense blob' problem.")
        logger.error("\n   Possible causes:")
        logger.error("   1. Sigmoid/padding bug in vae_model.py")
        logger.error("   2. Training data is too dense")
        logger.error("   3. Threshold too low (try --threshold 0.6)")
    elif np.mean(densities) < 0.01:
        logger.error("❌ Generated houses are nearly empty!")
        logger.error("   Threshold might be too high (try --threshold 0.4)")
    elif 0.1 <= np.mean(densities) <= 0.3:
        logger.info("✅ Generated density looks reasonable!")
    else:
        logger.warning(f"⚠️  Generated density ({np.mean(densities):.2%}) is unusual")

    # Check for diversity
    if np.std(densities) < 0.01:
        logger.warning("⚠️  All generated houses have similar density!")
        logger.warning("   Model might not have learned diverse structures.")
        logger.warning("   Try: lower kl_weight (--kl_weight 0.0005)")
    else:
        logger.info(f"✅ Good diversity (density std={np.std(densities):.3f})")


def main():
    """Run all diagnostic checks."""
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose VAE training issues')
    parser.add_argument('--houses_dir', type=str, default='houses',
                        help='Directory with training houses')
    parser.add_argument('--generated_dir', type=str, default='generated_houses',
                        help='Directory with generated houses')
    parser.add_argument('--skip-model-check', action='store_true',
                        help='Skip model architecture check')

    args = parser.parse_args()

    logger.info("🔍 VAE HOUSE GENERATOR - DIAGNOSTIC TOOL")
    logger.info("=" * 70)

    # Check training data
    data_ok = check_training_data(args.houses_dir)

    # Check generated samples (if they exist)
    check_generated_samples(args.generated_dir)

    # Final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'=' * 70}")

    if not data_ok:
        logger.error("❌ CRITICAL: Need more training data!")
        logger.info("\n   Next steps:")
        logger.info("   1. Collect 20-50 diverse houses")
        logger.info("   2. Save them as .litematic/.schem files")
        logger.info("   3. Place in houses/litematic/ or houses/schem/")
        logger.info("   4. Run this diagnostic again")
    else:
        logger.info("✅ Diagnostic complete!")


if __name__ == "__main__":
    main()
