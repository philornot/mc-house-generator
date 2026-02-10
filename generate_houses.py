"""Generate new Minecraft houses using trained VAE model.

This script loads a trained VAE and generates new house structures.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from logger import get_logger, setup_logging
from vae_model import VAE3D

logger = get_logger(__name__)


class HouseGenerator:
    """Generator for creating new houses using trained VAE."""

    def __init__(
            self,
            model: VAE3D,
            device: torch.device,
            threshold: float = 0.5
    ):
        """Initialize generator.

        Args:
            model: Trained VAE model.
            device: Device to generate on.
            threshold: Threshold for converting probabilities to binary (0-1).
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()

    @torch.no_grad()
    def generate_random(self, num_samples: int = 1) -> np.ndarray:
        """Generate random houses by sampling from latent space.

        Args:
            num_samples: Number of houses to generate.

        Returns:
            Binary voxel arrays [num_samples, X, Y, Z].
        """
        logger.info(f"Generating {num_samples} random houses...")

        # Sample from latent space
        samples = self.model.sample(num_samples, self.device)

        # Convert to binary
        samples_binary = (samples > self.threshold).float()

        # Remove channel dimension and convert to numpy
        samples_binary = samples_binary.squeeze(1).cpu().numpy()

        # Calculate statistics
        for i, sample in enumerate(samples_binary):
            density = sample.mean()
            logger.info(f"  House {i + 1}: density={density:.2%}, "
                        f"blocks={int(sample.sum())}")

        return samples_binary

    @torch.no_grad()
    def interpolate(
            self,
            start_latent: torch.Tensor,
            end_latent: torch.Tensor,
            num_steps: int = 10
    ) -> np.ndarray:
        """Generate houses by interpolating between two latent vectors.

        Args:
            start_latent: Starting latent vector [latent_dim].
            end_latent: Ending latent vector [latent_dim].
            num_steps: Number of interpolation steps.

        Returns:
            Binary voxel arrays [num_steps, X, Y, Z].
        """
        logger.info(f"Interpolating {num_steps} houses...")

        # Create interpolation weights
        alphas = np.linspace(0, 1, num_steps)

        houses = []
        for alpha in alphas:
            # Interpolate in latent space
            latent = (1 - alpha) * start_latent + alpha * end_latent
            latent = latent.unsqueeze(0)  # Add batch dimension

            # Decode
            house = self.model.decode(latent)
            house_binary = (house > self.threshold).float()
            house_binary = house_binary.squeeze().cpu().numpy()

            houses.append(house_binary)

        return np.array(houses)

    @torch.no_grad()
    def generate_variations(
            self,
            base_latent: torch.Tensor,
            num_variations: int = 5,
            noise_scale: float = 0.5
    ) -> np.ndarray:
        """Generate variations of a house by adding noise to latent vector.

        Args:
            base_latent: Base latent vector [latent_dim].
            num_variations: Number of variations to generate.
            noise_scale: Scale of noise to add (0-1).

        Returns:
            Binary voxel arrays [num_variations, X, Y, Z].
        """
        logger.info(f"Generating {num_variations} variations...")

        houses = []
        for i in range(num_variations):
            # Add noise
            noise = torch.randn_like(base_latent) * noise_scale
            latent = base_latent + noise
            latent = latent.unsqueeze(0)  # Add batch dimension

            # Decode
            house = self.model.decode(latent)
            house_binary = (house > self.threshold).float()
            house_binary = house_binary.squeeze().cpu().numpy()

            houses.append(house_binary)

            density = house_binary.mean()
            logger.info(f"  Variation {i + 1}: density={density:.2%}")

        return np.array(houses)

    def save_voxels(self, voxels: np.ndarray, output_path: str):
        """Save voxel array to numpy file.

        Args:
            voxels: Voxel array [X, Y, Z] or [N, X, Y, Z].
            output_path: Path to save file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, voxels)
        logger.info(f"Saved voxels to {output_path}")


def load_model(checkpoint_path: str, latent_dim: int, device: torch.device) -> VAE3D:
    """Load trained VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        latent_dim: Latent dimension of model.
        device: Device to load model on.

    Returns:
        Loaded VAE model.
    """
    model = VAE3D(latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Loaded model from checkpoint (epoch {epoch})")

    return model


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate houses using trained VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_houses',
                        help='Directory to save generated houses')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of houses to generate')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension (must match trained model)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary conversion (0-1)')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'interpolate', 'variations'],
                        help='Generation mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.latent_dim, device)

    # Create generator
    generator = HouseGenerator(model, device, threshold=args.threshold)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate houses based on mode
    if args.mode == 'random':
        # Generate random houses
        houses = generator.generate_random(num_samples=args.num_samples)

        # Save each house
        for i, house in enumerate(houses):
            output_path = output_dir / f"house_{i + 1:03d}.npy"
            generator.save_voxels(house, str(output_path))

    elif args.mode == 'interpolate':
        # Generate interpolation between two random points
        z_start = torch.randn(args.latent_dim).to(device)
        z_end = torch.randn(args.latent_dim).to(device)

        houses = generator.interpolate(z_start, z_end, num_steps=args.num_samples)

        # Save interpolation sequence
        output_path = output_dir / "interpolation_sequence.npy"
        generator.save_voxels(houses, str(output_path))

        # Also save individual houses
        for i, house in enumerate(houses):
            output_path = output_dir / f"interpolation_{i + 1:03d}.npy"
            generator.save_voxels(house, str(output_path))

    elif args.mode == 'variations':
        # Generate variations of a base house
        z_base = torch.randn(args.latent_dim).to(device)

        houses = generator.generate_variations(
            z_base,
            num_variations=args.num_samples,
            noise_scale=0.3
        )

        # Save each variation
        for i, house in enumerate(houses):
            output_path = output_dir / f"variation_{i + 1:03d}.npy"
            generator.save_voxels(house, str(output_path))

    logger.info(f"✅ Generation complete! Houses saved to {output_dir}")
    logger.info(f"To visualize, you can:")
    logger.info(f"  1. Load .npy files with numpy: np.load('house_001.npy')")
    logger.info(f"  2. Convert to Minecraft format (coming soon!)")


if __name__ == "__main__":
    main()
