"""Getting Started Guide - Interactive helper script.

Run this to get step-by-step instructions on using the house generator.
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    missing = []

    try:
        import numpy
        print("✅ numpy")
    except ImportError:
        missing.append("numpy")
        print("❌ numpy")

    try:
        import torch
        print("✅ torch")
    except ImportError:
        missing.append("torch")
        print("❌ torch")

    try:
        import nbtlib
        print("✅ nbtlib")
    except ImportError:
        missing.append("nbtlib")
        print("❌ nbtlib")

    try:
        import litemapy
        print("✅ litemapy")
    except ImportError:
        missing.append("litemapy")
        print("❌ litemapy")

    try:
        import matplotlib
        print("✅ matplotlib")
    except ImportError:
        missing.append("matplotlib")
        print("❌ matplotlib")

    try:
        import tqdm
        print("✅ tqdm")
    except ImportError:
        missing.append("tqdm")
        print("❌ tqdm")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall them with:")
        print("    pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def check_data():
    """Check if house data is available."""
    print("\nChecking for house data...")

    houses_dir = Path("houses")

    if not houses_dir.exists():
        print("❌ 'houses' directory not found")
        print("\nCreate it with:")
        print("    mkdir houses")
        print("    mkdir houses/litematic houses/schem houses/schematic")
        return False

    # Check for files
    total_files = 0

    for fmt, ext in [('litematic', '*.litematic'),
                     ('schem', '*.schem'),
                     ('schematic', '*.schematic')]:
        subdir = houses_dir / fmt
        if subdir.exists():
            files = list(subdir.glob(ext))
            if files:
                print(f"✅ {len(files)} {fmt} file(s) found")
                total_files += len(files)
            else:
                print(f"⚠️  No {fmt} files in {subdir}")
        else:
            print(f"⚠️  Directory {subdir} not found")

    if total_files == 0:
        print("\n❌ No house files found!")
        print("\nAdd schematic files to:")
        print("    houses/litematic/  - for .litematic files")
        print("    houses/schem/      - for .schem files")
        print("    houses/schematic/  - for .schematic files")
        print("\nYou can find schematics at:")
        print("    - https://www.minecraft-schematics.com/")
        print("    - https://www.planetminecraft.com/")
        return False

    print(f"\n✅ Found {total_files} house file(s)!")

    if total_files < 10:
        print("\n⚠️  Warning: You have less than 10 houses.")
        print("   For better results, try to get at least 20-50 diverse houses.")

    return True


def main():
    """Main function."""
    print_header("🏠 MINECRAFT HOUSE GENERATOR - GETTING STARTED 🏠")

    print("This guide will help you set up and use the house generator.\n")

    # Step 1: Check dependencies
    print_header("STEP 1: Check Dependencies")
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n❌ Please install dependencies first, then run this script again.")
        sys.exit(1)

    # Step 2: Check data
    print_header("STEP 2: Check House Data")
    data_ok = check_data()

    if not data_ok:
        print("\n❌ Please add house files, then run this script again.")
        sys.exit(1)

    # Step 3: Next steps
    print_header("STEP 3: What to do next?")

    print("Everything is ready! Here's what you can do:\n")

    print("🔍 TEST YOUR SETUP:")
    print("    python test_vae_model.py")
    print("    python test_house_dataset.py\n")

    print("🎓 START TRAINING (Quick test - 50 epochs):")
    print("    python train_vae.py --epochs 50 --batch_size 4\n")

    print("🎓 FULL TRAINING (Recommended - 100-200 epochs):")
    print("    python train_vae.py --epochs 100 --batch_size 4 --kl_weight 0.001\n")

    print("📊 MONITOR TRAINING:")
    print("    tensorboard --logdir runs")
    print("    # Then open http://localhost:6006 in browser\n")

    print("🎨 GENERATE HOUSES (after training):")
    print("    python generate_houses.py --checkpoint checkpoints/best_model.pth --num_samples 10\n")

    print("👀 VISUALIZE GENERATED HOUSES:")
    print("    python visualize_houses.py generated_houses/house_001.npy")
    print("    python visualize_houses.py generated_houses/ --max_houses 6\n")

    print_header("💡 TIPS FOR SUCCESS")

    print("📈 Training tips:")
    print("  - Start with 50-100 epochs to see if it's working")
    print("  - Monitor reconstruction loss - should decrease steadily")
    print("  - If generated houses are empty/full, adjust kl_weight")
    print("  - More training data = better results!\n")

    print("🎯 Dataset tips:")
    print("  - 10-20 houses: Will work, but limited variety")
    print("  - 50-100 houses: Good results, diverse generations")
    print("  - 100+ houses: Best results!")
    print("  - Make sure houses are diverse (different styles/sizes)\n")

    print("⚙️ Hyperparameter tips:")
    print("  - kl_weight: 0.0005-0.005 (most important!)")
    print("  - learning_rate: 0.001 (default is good)")
    print("  - batch_size: 2-4 (depending on GPU memory)")
    print("  - latent_dim: 256 (default is good)\n")

    print_header("📚 NEED HELP?")

    print("📖 Read the README.md for detailed instructions")
    print("🐛 Check the Troubleshooting section if something goes wrong")
    print("💬 The code is well-commented - read through it!\n")

    print_header("🚀 READY TO START!")

    print("Run this command to start training:\n")
    print("    python train_vae.py --epochs 100 --batch_size 4\n")
    print("Good luck! 🎉\n")


if __name__ == "__main__":
    main()