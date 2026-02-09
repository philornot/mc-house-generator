"""Visualize generated Minecraft houses in 3D.

Simple visualization tool to preview generated voxel structures.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class VoxelVisualizer:
    """Visualizer for 3D voxel structures."""

    @staticmethod
    def visualize_single(
            voxels: np.ndarray,
            title: str = "House",
            show: bool = True,
            save_path: str = None
    ):
        """Visualize a single voxel structure.

        Args:
            voxels: Binary voxel array [X, Y, Z].
            title: Title for the plot.
            show: Whether to display the plot.
            save_path: Optional path to save the figure.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get coordinates of filled voxels
        filled = voxels > 0.5
        x, y, z = np.where(filled)

        # Plot voxels
        ax.scatter(x, y, z, c='brown', marker='s', s=20, alpha=0.6)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}\nBlocks: {filled.sum()}, Density: {filled.mean():.2%}")

        # Set equal aspect ratio
        max_range = max(voxels.shape)
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")

        if show:
            plt.show()

        plt.close()

    @staticmethod
    def visualize_slices(
            voxels: np.ndarray,
            title: str = "House Slices",
            show: bool = True,
            save_path: str = None
    ):
        """Visualize voxel structure as 2D slices.

        Args:
            voxels: Binary voxel array [X, Y, Z].
            title: Title for the plot.
            show: Whether to display the plot.
            save_path: Optional path to save the figure.
        """
        # Take slices along Y axis (horizontal slices)
        num_slices = min(8, voxels.shape[1])
        slice_indices = np.linspace(0, voxels.shape[1] - 1, num_slices, dtype=int)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, slice_idx in enumerate(slice_indices):
            slice_2d = voxels[:, slice_idx, :]

            axes[i].imshow(slice_2d.T, cmap='binary', origin='lower')
            axes[i].set_title(f'Y={slice_idx}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved slice visualization to {save_path}")

        if show:
            plt.show()

        plt.close()

    @staticmethod
    def visualize_multiple(
            voxels_list: list,
            titles: list = None,
            save_path: str = None
    ):
        """Visualize multiple voxel structures side by side.

        Args:
            voxels_list: List of voxel arrays.
            titles: List of titles for each structure.
            save_path: Optional path to save the figure.
        """
        num_houses = len(voxels_list)
        cols = min(3, num_houses)
        rows = (num_houses + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows))

        for i, voxels in enumerate(voxels_list):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

            # Get coordinates of filled voxels
            filled = voxels > 0.5
            x, y, z = np.where(filled)

            # Plot voxels
            ax.scatter(x, y, z, c='brown', marker='s', s=10, alpha=0.6)

            # Set title
            title = titles[i] if titles and i < len(titles) else f"House {i + 1}"
            ax.set_title(f"{title}\nBlocks: {filled.sum()}")

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set equal aspect ratio
            max_range = max(voxels.shape)
            ax.set_xlim([0, max_range])
            ax.set_ylim([0, max_range])
            ax.set_zlim([0, max_range])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved multi-visualization to {save_path}")

        plt.show()
        plt.close()

    @staticmethod
    def visualize_multiple_separate(
            voxels_list: list,
            titles: list = None,
            delay: float = 0.1
    ):
        """Visualize multiple houses as SEPARATE plots (PyCharm-friendly).

        Each house gets its own figure window / PyCharm plot.
        Perfect for PyCharm's Plots panel!

        Args:
            voxels_list: List of voxel arrays.
            titles: List of titles for each structure.
            delay: Delay between plots (seconds) to ensure PyCharm captures each.
        """
        import time

        logger.info(f"Creating {len(voxels_list)} separate plots...")
        logger.info("Each plot will appear in PyCharm's Plots panel")

        for i, voxels in enumerate(voxels_list):
            # Create NEW figure for each house
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Get coordinates of filled voxels
            filled = voxels > 0.5
            x, y, z = np.where(filled)

            # Plot voxels
            ax.scatter(x, y, z, c='brown', marker='s', s=20, alpha=0.6)

            # Set title
            title = titles[i] if titles and i < len(titles) else f"House {i + 1}"
            density = filled.mean()
            ax.set_title(f"{title}\nBlocks: {filled.sum()}, Density: {density:.2%}",
                         fontsize=14, fontweight='bold')

            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)

            # Set equal aspect ratio
            max_range = max(voxels.shape)
            ax.set_xlim([0, max_range])
            ax.set_ylim([0, max_range])
            ax.set_zlim([0, max_range])

            plt.tight_layout()

            # IMPORTANT: plt.show() creates separate plot in PyCharm
            plt.show()

            # Small delay to ensure PyCharm captures the plot
            if i < len(voxels_list) - 1:
                time.sleep(delay)

            logger.info(f"  Plot {i + 1}/{len(voxels_list)}: {title}")

        logger.info(f"✅ Created {len(voxels_list)} separate plots!")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize generated houses')
    parser.add_argument('input', type=str,
                        help='Path to .npy file or directory with .npy files')
    parser.add_argument('--mode', type=str, default='3d',
                        choices=['3d', 'slices', 'both'],
                        help='Visualization mode')
    parser.add_argument('--max_houses', type=int, default=100,
                        help='Maximum number of houses to visualize from directory')
    parser.add_argument('--save', action='store_true',
                        help='Save visualizations instead of showing')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save each house as individual file (useful for many houses)')
    parser.add_argument('--pycharm', action='store_true',
                        help='PyCharm mode - create separate plot for each house (appears in Plots panel)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations (if --save is used)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    input_path = Path(args.input)

    # Load voxel data
    if input_path.is_file():
        # Single file
        logger.info(f"Loading {input_path}")
        voxels = np.load(input_path)

        if voxels.ndim == 4:
            # Multiple houses in one file
            logger.info(f"Found {len(voxels)} houses in file")
            voxels_list = [voxels[i] for i in range(min(len(voxels), args.max_houses))]
        else:
            voxels_list = [voxels]

    elif input_path.is_dir():
        # Directory with multiple files
        npy_files = sorted(list(input_path.glob('*.npy')))
        logger.info(f"Found {len(npy_files)} .npy files")

        voxels_list = []
        for npy_file in npy_files[:args.max_houses]:
            logger.info(f"Loading {npy_file.name}")
            voxels = np.load(npy_file)
            if voxels.ndim == 3:
                voxels_list.append(voxels)
            elif voxels.ndim == 4:
                # Take first house if multiple in file
                voxels_list.append(voxels[0])

    else:
        logger.error(f"Input path {input_path} not found")
        return

    if not voxels_list:
        logger.error("No valid voxel data found")
        return

    logger.info(f"Loaded {len(voxels_list)} houses")

    # Visualize
    visualizer = VoxelVisualizer()

    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if len(voxels_list) == 1:
        # Single house
        voxels = voxels_list[0]

        if args.mode in ['3d', 'both']:
            save_path = None
            if args.save:
                save_path = output_dir / 'house_3d.png'
            visualizer.visualize_single(
                voxels,
                title="Generated House",
                show=not args.save,
                save_path=save_path
            )

        if args.mode in ['slices', 'both']:
            save_path = None
            if args.save:
                save_path = output_dir / 'house_slices.png'
            visualizer.visualize_slices(
                voxels,
                title="Generated House - Slices",
                show=not args.save,
                save_path=save_path
            )

    else:
        # Multiple houses
        if args.pycharm:
            # PyCharm mode - separate plot for each house
            titles = [f"House {i + 1}" for i in range(len(voxels_list))]
            visualizer.visualize_multiple_separate(
                voxels_list,
                titles=titles
            )

        elif args.save_individual:
            # Save each house individually
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving {len(voxels_list)} houses individually...")

            for i, voxels in enumerate(voxels_list):
                if args.mode in ['3d', 'both']:
                    save_path = output_dir / f'house_{i + 1:03d}_3d.png'
                    visualizer.visualize_single(
                        voxels,
                        title=f"House {i + 1}",
                        show=False,
                        save_path=save_path
                    )

                if args.mode in ['slices', 'both']:
                    save_path = output_dir / f'house_{i + 1:03d}_slices.png'
                    visualizer.visualize_slices(
                        voxels,
                        title=f"House {i + 1} - Slices",
                        show=False,
                        save_path=save_path
                    )

            logger.info(f"✅ Saved all houses to {output_dir}")

        else:
            # Show/save comparison view
            save_path = None
            if args.save:
                save_path = output_dir / 'houses_comparison.png'

            titles = [f"House {i + 1}" for i in range(len(voxels_list))]
            visualizer.visualize_multiple(
                voxels_list,
                titles=titles,
                save_path=save_path
            )

    logger.info("✅ Visualization complete!")


if __name__ == "__main__":
    main()
