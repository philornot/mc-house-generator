"""
Training script for 3D VAE.
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
    """Trainer for 3D VAE model with KL annealing and gradient clipping."""

    def __init__(
            self,
            model: VAE3D,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            optimizer: optim.Optimizer,
            device: torch.device,
            checkpoint_dir: str = "checkpoints",
            log_dir: str = "runs",
            kl_weight: float = 0.001,
            physics_weight: float = 0.5,
            connectivity_weight: float = 0.3,
            ground_weight: float = 0.2,
            symmetry_weight: float = 0.1,
            vertical_weight: float = 0.1,
            warmup_epochs: int = 30,
            max_grad_norm: float = 1.0,
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
            kl_weight: Target weight for KL divergence loss term.
            physics_weight: Weight for physics constraints.
            connectivity_weight: Weight for connectivity constraints.
            ground_weight: Weight for ground plane constraints.
            symmetry_weight: Weight for symmetry encouragement.
            vertical_weight: Weight for vertical structure.
            warmup_epochs: Number of epochs to anneal kl_weight from 0 to target.
            max_grad_norm: Gradient clipping max norm (prevents KL explosion).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.target_kl_weight = kl_weight
        self.physics_weight = physics_weight
        self.connectivity_weight = connectivity_weight
        self.ground_weight = ground_weight
        self.symmetry_weight = symmetry_weight
        self.vertical_weight = vertical_weight
        self.warmup_epochs = warmup_epochs
        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_recon = float('inf')

        logger.info("Constraint weights:")
        logger.info(f"  KL target: {kl_weight}  (annealed over {warmup_epochs} epochs)")
        logger.info(f"  Physics: {physics_weight}")
        logger.info(f"  Connectivity: {connectivity_weight}")
        logger.info(f"  Ground: {ground_weight}")
        logger.info(f"  Symmetry: {symmetry_weight}")
        logger.info(f"  Vertical: {vertical_weight}")
        logger.info(f"  Gradient clip norm: {max_grad_norm}")

    def _current_kl_weight(self) -> float:
        """Compute annealed KL weight for the current epoch.

        Linearly ramps from 0 to target_kl_weight over warmup_epochs.

        Returns:
            Current effective KL weight.
        """
        if self.warmup_epochs <= 0:
            return self.target_kl_weight
        progress = min(1.0, self.current_epoch / self.warmup_epochs)
        return self.target_kl_weight * progress

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with per-component average losses for the epoch.
        """
        self.model.train()
        kl_weight = self._current_kl_weight()

        accumulators = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'physics_loss': 0.0,
            'connectivity_loss': 0.0,
            'ground_loss': 0.0,
            'symmetry_loss': 0.0,
            'vertical_loss': 0.0,
        }
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (kl_w={kl_weight:.5f})",
            leave=False,
            mininterval=0.5,
        )

        for batch in pbar:
            voxels = batch['voxels'].to(self.device)

            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(voxels)

            losses = vae_loss(
                reconstruction, voxels, mu, logvar,
                kl_weight=kl_weight,
                physics_weight=self.physics_weight,
                connectivity_weight=self.connectivity_weight,
                ground_weight=self.ground_weight,
                symmetry_weight=self.symmetry_weight,
                vertical_weight=self.vertical_weight,
            )

            losses['total_loss'].backward()

            # FIX: clip gradients to prevent single bad batch from blowing up KL
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            self.optimizer.step()

            for key in accumulators:
                accumulators[key] += losses[key].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.1f}",
                'recon': f"{losses['reconstruction_loss'].item():.1f}",
                'KL': f"{losses['kl_loss'].item():.1f}",
                'phys': f"{losses['physics_loss'].item():.4f}",
            })

        return {k: v / num_batches for k, v in accumulators.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on the validation set.

        Returns:
            Dictionary with per-component average validation losses.
            Empty dict if no val_loader.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        kl_weight = self._current_kl_weight()

        accumulators = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'physics_loss': 0.0,
            'connectivity_loss': 0.0,
            'ground_loss': 0.0,
            'symmetry_loss': 0.0,
            'vertical_loss': 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            voxels = batch['voxels'].to(self.device)
            reconstruction, mu, logvar = self.model(voxels)

            losses = vae_loss(
                reconstruction, voxels, mu, logvar,
                kl_weight=kl_weight,
                physics_weight=self.physics_weight,
                connectivity_weight=self.connectivity_weight,
                ground_weight=self.ground_weight,
                symmetry_weight=self.symmetry_weight,
                vertical_weight=self.vertical_weight,
            )

            for key in accumulators:
                accumulators[key] += losses[key].item()
            num_batches += 1

        return {k: v / num_batches for k, v in accumulators.items()}

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file (placed inside checkpoint_dir).
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_recon': self.best_val_recon,
            'latent_dim': self.model.latent_dim,
            'input_shape': self.model.input_shape,
            'kl_weight': self.target_kl_weight,
            'physics_weight': self.physics_weight,
            'connectivity_weight': self.connectivity_weight,
            'ground_weight': self.ground_weight,
            'symmetry_weight': self.symmetry_weight,
            'vertical_weight': self.vertical_weight,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Name of checkpoint file inside checkpoint_dir.
        """
        path = self.checkpoint_dir / filename
        if not path.exists():
            logger.warning(f"Checkpoint {path} not found")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_recon = checkpoint.get('best_val_recon', float('inf'))
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def train(self, num_epochs: int, save_every: int = 10):
        """Train model for a given number of epochs.

        Args:
            num_epochs: Total number of epochs to run.
            save_every: Save a periodic checkpoint every N epochs.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()

            for key, value in train_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            self.writer.add_scalar('train/kl_weight', self._current_kl_weight(), epoch)

            logger.info(
                f"Epoch {epoch} - Train Loss: {train_losses['total_loss']:.4f} "
                f"(Recon: {train_losses['reconstruction_loss']:.4f}, "
                f"KL: {train_losses['kl_loss']:.4f}, "
                f"Physics: {train_losses['physics_loss']:.4f}, "
                f"Connect: {train_losses['connectivity_loss']:.4f})"
            )

            if self.val_loader:
                val_losses = self.validate()

                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)

                logger.info(
                    f"Epoch {epoch} - Val Loss: {val_losses['total_loss']:.4f} "
                    f"(Recon: {val_losses['reconstruction_loss']:.4f}, "
                    f"Physics: {val_losses['physics_loss']:.4f})"
                )

                # Save best-total-loss checkpoint
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint('best_model.pth')
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

                # Save best-reconstruction checkpoint separately
                # (more reliable than total loss during warmup)
                if val_losses['reconstruction_loss'] < self.best_val_recon:
                    self.best_val_recon = val_losses['reconstruction_loss']
                    self.save_checkpoint('best_recon_model.pth')
                    logger.info(
                        f"New best val recon: {self.best_val_recon:.4f} "
                        f"→ saved best_recon_model.pth"
                    )

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            if (epoch + 1) % 5 == 0:
                self._log_sample(epoch)

        self.save_checkpoint('final_model.pth')
        logger.info("Training completed!")
        self.writer.close()

    @torch.no_grad()
    def _log_sample(self, epoch: int):
        """Generate one sample and log metrics to tensorboard.

        Args:
            epoch: Current epoch number (used as x-axis in tensorboard).
        """
        self.model.eval()

        sample = self.model.sample(num_samples=1, device=self.device)
        sample_binary = (sample > 0.5).float()

        density = sample_binary.mean().item()
        self.writer.add_scalar('sample/density', density, epoch)

        from vae_model import physics_loss, connectivity_loss, ground_plane_loss

        phys = physics_loss(sample).item()
        conn = connectivity_loss(sample).item()
        gnd = ground_plane_loss(sample).item()

        self.writer.add_scalar('sample/physics_violations', phys, epoch)
        self.writer.add_scalar('sample/isolated_blocks', conn, epoch)
        self.writer.add_scalar('sample/ground_contact', gnd, epoch)

        logger.info(
            f"Generated sample - Density: {density:.2%}, "
            f"Physics: {phys:.4f}, Connectivity: {conn:.4f}"
        )


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train 3D VAE with architectural constraints'
    )
    parser.add_argument('--houses_dir', type=str, default='houses')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--kl_weight', type=float, default=0.001,
                        help='Target KL weight (annealed during warmup)')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='Epochs to anneal kl_weight from 0 to target')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--physics_weight', type=float, default=0.5)
    parser.add_argument('--connectivity_weight', type=float, default=0.3)
    parser.add_argument('--ground_weight', type=float, default=0.2)
    parser.add_argument('--symmetry_weight', type=float, default=0.1)
    parser.add_argument('--vertical_weight', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    setup_logging()

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    logger.info(f"Loading dataset from {args.houses_dir}")
    full_dataset = HouseVoxelDataset(
        houses_dir=args.houses_dir,
        binary_mode=True,
        max_size=None,
    )

    stats = full_dataset.get_statistics()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    if args.val_split > 0:
        val_size = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info(f"Split dataset: {train_size} train, {val_size} validation")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info("No validation split")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == 'cuda'),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(args.device == 'cuda'),
        )

    logger.info(f"Creating VAE with latent_dim={args.latent_dim}")
    input_shape = full_dataset.max_size
    logger.info(f"Model will use input shape: {input_shape}")

    model = VAE3D(latent_dim=args.latent_dim, input_shape=input_shape).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        kl_weight=args.kl_weight,
        physics_weight=args.physics_weight,
        connectivity_weight=args.connectivity_weight,
        ground_weight=args.ground_weight,
        symmetry_weight=args.symmetry_weight,
        vertical_weight=args.vertical_weight,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(num_epochs=args.epochs, save_every=10)


if __name__ == "__main__":
    main()
