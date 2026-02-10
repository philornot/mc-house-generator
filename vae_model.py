"""3D Variational Autoencoder for Minecraft house generation - FIXED VERSION.

FIXES:
1. Sigmoid applied BEFORE padding (critical bug fix)
2. Added comments explaining the architecture
3. Better handling of dimension mismatches

This VAE works with binary voxel data (0=air, 1=block) and learns to
generate new house structures by learning a compressed latent representation.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder3D(nn.Module):
    """3D Convolutional Encoder for voxel data.

    Compresses 3D voxel grid into latent representation.
    """

    def __init__(self, latent_dim: int = 256):
        """Initialize encoder.

        Args:
            latent_dim: Dimension of latent space (both mu and logvar).
        """
        super().__init__()

        # Input: [B, 1, 80, 42, 80]
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)  # [B, 32, 40, 21, 40]
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # [B, 64, 20, 10, 20]
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # [B, 128, 10, 5, 10]
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # [B, 256, 5, 2, 5]
        self.bn4 = nn.BatchNorm3d(256)

        # Calculate flattened size: 256 * 5 * 2 * 5 = 12,800
        self.flatten_size = 256 * 5 * 2 * 5

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Args:
            x: Input voxel tensor [B, 1, 80, 42, 80].

        Returns:
            Tuple of (mu, logvar) both [B, latent_dim].
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder3D(nn.Module):
    """3D Convolutional Decoder for voxel data.

    Reconstructs 3D voxel grid from latent representation.

    FIXED: Sigmoid is now applied BEFORE padding to avoid the critical bug
    where F.pad(x, value=0) followed by sigmoid(x) gives 0.5 instead of 0.
    """

    def __init__(self, latent_dim: int = 256):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
        """
        super().__init__()

        # Project latent vector to 3D feature map
        self.flatten_size = 256 * 5 * 2 * 5
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Transposed convolutions to upsample
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)  # [B, 128, 10, 4, 10]
        self.bn1 = nn.BatchNorm3d(128)

        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)  # [B, 64, 20, 8, 20]
        self.bn2 = nn.BatchNorm3d(64)

        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)  # [B, 32, 40, 16, 40]
        self.bn3 = nn.BatchNorm3d(32)

        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)  # [B, 1, 80, 32, 80]

        # We need to adjust to get exactly [80, 42, 80]
        # Current output: [80, 32, 80], need [80, 42, 80]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed voxel tensor [B, 1, 80, 42, 80].
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), 256, 5, 2, 5)

        # Upsample
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)

        # CRITICAL FIX: Apply sigmoid BEFORE padding!
        # This ensures padded values are 0 (air), not 0.5 (ambiguous)
        x = torch.sigmoid(x)

        # Adjust Y dimension from 32 to 42 by padding
        # Since we already applied sigmoid, padded zeros stay as 0 (air)
        if x.size(3) != 42:
            pad_y = 42 - x.size(3)
            x = F.pad(x, (0, 0, 0, pad_y, 0, 0), value=0.0)

        return x


class VAE3D(nn.Module):
    """3D Variational Autoencoder for house generation.

    Combines encoder and decoder with reparameterization trick.
    """

    def __init__(self, latent_dim: int = 256):
        """Initialize VAE.

        Args:
            latent_dim: Dimension of latent space.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].

        Returns:
            Sampled latent vector [B, latent_dim].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input voxel tensor [B, 1, 80, 42, 80].

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated voxel tensors [num_samples, 1, 80, 42, 80].
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Decode
        with torch.no_grad():
            samples = self.decoder(z)

        return samples

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (using mean, not sampling).

        Args:
            x: Input voxel tensor [B, 1, 80, 42, 80].

        Returns:
            Latent representation [B, latent_dim].
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to voxel space.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed voxel tensor [B, 1, 80, 42, 80].
        """
        return self.decoder(z)


def vae_loss(
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.001
) -> Dict[str, torch.Tensor]:
    """Calculate VAE loss (reconstruction + KL divergence).

    Args:
        reconstruction: Reconstructed output [B, 1, 80, 42, 80].
        x: Original input [B, 1, 80, 42, 80].
        mu: Latent mean [B, latent_dim].
        logvar: Latent log variance [B, latent_dim].
        kl_weight: Weight for KL divergence term (important for balancing).

    Returns:
        Dictionary with 'total_loss', 'reconstruction_loss', 'kl_loss'.
    """
    # Binary cross-entropy for reconstruction (treating as binary classification)
    bce_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    bce_loss = bce_loss / x.size(0)  # Average over batch

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch

    # Total loss
    total_loss = bce_loss + kl_weight * kl_loss

    return {
        'total_loss': total_loss,
        'reconstruction_loss': bce_loss,
        'kl_loss': kl_loss
    }


def main():
    """Test VAE architecture."""
    print("Testing 3D VAE architecture...")

    # Create model
    model = VAE3D(latent_dim=256)

    # Test with random input
    batch_size = 4
    x = torch.randn(batch_size, 1, 80, 42, 80)

    # Forward pass
    reconstruction, mu, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Test loss
    losses = vae_loss(reconstruction, x, mu, logvar)
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    # Test sampling
    samples = model.sample(num_samples=2, device=x.device)
    print(f"\nGenerated samples shape: {samples.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✅ VAE architecture test passed!")


if __name__ == "__main__":
    main()
