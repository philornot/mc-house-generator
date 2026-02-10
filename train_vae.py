"""Training script for 3D VAE house generator.

This script trains the VAE model on the house dataset and saves checkpoints.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from house_dataset import HouseVoxelDataset
from logger import get_logger, setup_logging
from vae_model import VAE3D, vae_loss

logger = get_logger(__name__)


class VAETrainer:
    """Trainer for 3D VAE model."""

    def __init__(
            self,
            model: VAE3D,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            optimizer: optim.Optimizer,
            device: torch.device,
            checkpoint_dir: str = "checkpoints",
            log_dir: str = "runs",
            kl_weight: float = 0.001
    ):
        """Initialize trainer.

        Args:
            model: VAE model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            optimizer: Optimizer for training.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory for tensorboard logs.
            kl_weight: Weight for KL divergence loss term.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.kl_weight = kl_weight

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with average losses for the epoch.
        """
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}",
                    leave=False, mininterval=0.5)

        for batch in pbar:
            voxels = batch['voxels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(voxels)

            # Calculate loss
            losses = vae_loss(reconstruction, voxels, mu, logvar, self.kl_weight)

            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['reconstruction_loss'].item(),
                'kl': losses['kl_loss'].item()
            })

        # Calculate averages
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set.

        Returns:
            Dictionary with average validation losses.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            voxels = batch['voxels'].to(self.device)

            # Forward pass
            reconstruction, mu, logvar = self.model(voxels)

            # Calculate loss
            losses = vae_loss(reconstruction, voxels, mu, logvar, self.kl_weight)

            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1

        # Calculate averages
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }

        return avg_losses

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'kl_weight': self.kl_weight
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return

        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

    def train(self, num_epochs: int, save_every: int = 10):
        """Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train.
            save_every: Save checkpoint every N epochs.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Log training losses
            for key, value in train_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

            logger.info(f"Epoch {epoch} - Train Loss: {train_losses['total_loss']:.4f} "
                        f"(Recon: {train_losses['reconstruction_loss']:.4f}, "
                        f"KL: {train_losses['kl_loss']:.4f})")

            # Validate
            if self.val_loader:
                val_losses = self.validate()

                # Log validation losses
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)

                logger.info(f"Epoch {epoch} - Val Loss: {val_losses['total_loss']:.4f} "
                            f"(Recon: {val_losses['reconstruction_loss']:.4f}, "
                            f"KL: {val_losses['kl_loss']:.4f})")

                # Save best model
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint('best_model.pth')
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Generate and log sample
            if (epoch + 1) % 5 == 0:
                self._log_sample(epoch)

        # Save final model
        self.save_checkpoint('final_model.pth')
        logger.info("Training completed!")

        self.writer.close()

    @torch.no_grad()
    def _log_sample(self, epoch: int):
        """Generate and log a sample to tensorboard.

        Args:
            epoch: Current epoch number.
        """
        self.model.eval()

        # Generate sample
        sample = self.model.sample(num_samples=1, device=self.device)

        # Convert to binary (threshold at 0.5)
        sample_binary = (sample > 0.5).float()

        # Log density
        density = sample_binary.mean().item()
        self.writer.add_scalar('sample/density', density, epoch)

        logger.info(f"Generated sample density: {density:.2%}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train 3D VAE for house generation')
    parser.add_argument('--houses_dir', type=str, default='houses',
                        help='Directory with house files')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=0.001,
                        help='Weight for KL divergence loss')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (0.0-1.0)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load dataset
    logger.info(f"Loading dataset from {args.houses_dir}")
    full_dataset = HouseVoxelDataset(
        houses_dir=args.houses_dir,
        binary_mode=True,
        max_size=None  # Auto-detect
    )

    # Show dataset stats
    stats = full_dataset.get_statistics()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Split dataset
    if args.val_split > 0:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Split dataset: {train_size} train, {val_size} validation")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info(f"No validation split, using all {len(train_dataset)} samples for training")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == 'cuda')
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(args.device == 'cuda')
        )

    # Create model
    logger.info(f"Creating VAE with latent_dim={args.latent_dim}")
    model = VAE3D(latent_dim=args.latent_dim).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        kl_weight=args.kl_weight
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_epochs=args.epochs, save_every=10)


if __name__ == "__main__":
    main()
