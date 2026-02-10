"""3D Variational Autoencoder with Architectural Constraints for Minecraft houses.

IMPROVEMENTS OVER BASIC VAE:
1. Physics constraints - blocks need support underneath
2. Connectivity constraints - no floating disconnected pieces
3. Ground plane constraints - houses sit on ground
4. Symmetry encouragement - houses tend to be symmetric
5. Vertical structure - encourage proper floors/walls/roof

These constraints help the model learn realistic house structures instead of random blobs.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder3D(nn.Module):
    """3D Convolutional Encoder for voxel data.

    Compresses 3D voxel grid into latent representation.
    Works with any input size (dynamically calculates flatten size).
    """

    def __init__(self, latent_dim: int = 256, input_shape: Tuple[int, int, int] = (80, 42, 80)):
        """Initialize encoder.

        Args:
            latent_dim: Dimension of latent space (both mu and logvar).
            input_shape: Expected input shape (X, Y, Z).
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Calculate flattened size dynamically
        self.flatten_size = self._get_flatten_size(input_shape)

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate the flattened size after conv layers.

        Args:
            input_shape: Input shape (X, Y, Z).

        Returns:
            Flattened size.
        """
        # Simulate forward pass to get output size
        x = torch.zeros(1, 1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Args:
            x: Input voxel tensor [B, 1, X, Y, Z].

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
    Works with any output size.
    """

    def __init__(self, latent_dim: int = 256, output_shape: Tuple[int, int, int] = (80, 42, 80)):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
            output_shape: Target output shape (X, Y, Z).
        """
        super().__init__()

        self.output_shape = output_shape

        # Calculate the size after encoder's conv layers
        # After 4 stride-2 convs: size // 16
        self.feature_x = max(1, output_shape[0] // 16)
        self.feature_y = max(1, output_shape[1] // 16)
        self.feature_z = max(1, output_shape[2] // 16)

        self.flatten_size = 256 * self.feature_x * self.feature_y * self.feature_z

        # Project latent vector to 3D feature map
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Transposed convolutions to upsample
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(128)

        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(32)

        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed voxel tensor [B, 1, X, Y, Z].
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.feature_x, self.feature_y, self.feature_z)

        # Upsample
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)

        # Apply sigmoid BEFORE any adjustments
        x = torch.sigmoid(x)

        # Adjust to exact target shape if needed
        current_shape = x.shape[2:]  # (X, Y, Z)
        target_shape = self.output_shape

        # Crop or pad each dimension
        for dim_idx, (current, target) in enumerate(zip(current_shape, target_shape)):
            if current > target:
                # Crop
                if dim_idx == 0:  # X
                    x = x[:, :, :target, :, :]
                elif dim_idx == 1:  # Y
                    x = x[:, :, :, :target, :]
                else:  # Z
                    x = x[:, :, :, :, :target]
            elif current < target:
                # Pad
                pad_amount = target - current
                if dim_idx == 0:  # X
                    x = F.pad(x, (0, 0, 0, 0, 0, pad_amount), value=0.0)
                elif dim_idx == 1:  # Y
                    x = F.pad(x, (0, 0, 0, pad_amount, 0, 0), value=0.0)
                else:  # Z
                    x = F.pad(x, (0, pad_amount, 0, 0, 0, 0), value=0.0)

        return x


class VAE3D(nn.Module):
    """3D Variational Autoencoder with architectural constraints."""

    def __init__(self, latent_dim: int = 256, input_shape: Tuple[int, int, int] = (80, 42, 80)):
        """Initialize VAE.

        Args:
            latent_dim: Dimension of latent space.
            input_shape: Expected input/output shape (X, Y, Z).
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = Encoder3D(latent_dim, input_shape)
        self.decoder = Decoder3D(latent_dim, input_shape)

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
        z = torch.randn(num_samples, self.latent_dim).to(device)

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


def physics_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalize floating blocks (blocks without support underneath).

    This encourages physically plausible structures where blocks
    have support from blocks below them.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] (after sigmoid, 0-1 range).
        threshold: Threshold to consider block as solid.

    Returns:
        Physics loss (scalar).
    """
    # Binarize
    binary = (voxels > threshold).float()

    # Get blocks that are not on ground level (Y > 0)
    blocks_above_ground = binary[:, :, :, 1:, :]

    # Check if there's a block directly below
    blocks_below = binary[:, :, :, :-1, :]

    # Blocks floating in air (no support below)
    floating = blocks_above_ground * (1 - blocks_below)

    return floating.mean()


def connectivity_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalize isolated blocks (blocks with no neighbors).

    Encourages connected structures instead of scattered blocks.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] (after sigmoid, 0-1 range).
        threshold: Threshold to consider block as solid.

    Returns:
        Connectivity loss (scalar).
    """
    # Binarize
    binary = (voxels > threshold).float()

    # Check 6 neighbors (3D connectivity)
    neighbors = torch.zeros_like(binary)

    # +X direction
    neighbors[:, :, :-1, :, :] += binary[:, :, 1:, :, :]
    # -X direction
    neighbors[:, :, 1:, :, :] += binary[:, :, :-1, :, :]
    # +Y direction
    neighbors[:, :, :, :-1, :] += binary[:, :, :, 1:, :]
    # -Y direction
    neighbors[:, :, :, 1:, :] += binary[:, :, :, :-1, :]
    # +Z direction
    neighbors[:, :, :, :, :-1] += binary[:, :, :, :, 1:]
    # -Z direction
    neighbors[:, :, :, :, 1:] += binary[:, :, :, :, :-1]

    # Penalize blocks with no neighbors (except if density is very low)
    isolated = binary * (neighbors == 0).float()

    # Only penalize if there are enough blocks
    if binary.sum() > 10:
        return isolated.mean()
    return torch.tensor(0.0, device=voxels.device)


def ground_plane_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Encourage blocks at ground level (Y=0).

    Houses should sit on the ground, not float in midair.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] (after sigmoid, 0-1 range).
        threshold: Threshold to consider block as solid.

    Returns:
        Ground plane loss (scalar).
    """
    # Binarize
    binary = (voxels > threshold).float()

    # Count blocks at ground level
    ground_blocks = binary[:, :, :, 0, :].sum()

    # Count total blocks
    total_blocks = binary.sum()

    # Encourage at least 5% of blocks on ground
    if total_blocks > 0:
        ground_ratio = ground_blocks / total_blocks
        target_ratio = 0.05
        return F.relu(target_ratio - ground_ratio)

    return torch.tensor(0.0, device=voxels.device)


def symmetry_loss(voxels: torch.Tensor) -> torch.Tensor:
    """Encourage symmetric structures along Z axis.

    Many houses are symmetric, this helps learn that pattern.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] (after sigmoid, 0-1 range).

    Returns:
        Symmetry loss (scalar).
    """
    # Flip along Z axis
    flipped = torch.flip(voxels, dims=[4])

    # MSE between original and flipped
    return F.mse_loss(voxels, flipped)


def vertical_structure_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Encourage vertical continuity (walls, pillars).

    Houses have walls that go from ground to roof, not random blocks.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] (after sigmoid, 0-1 range).
        threshold: Threshold to consider block as solid.

    Returns:
        Vertical structure loss (scalar).
    """
    # Binarize
    binary = (voxels > threshold).float()

    # For each X-Z position, check if blocks form continuous vertical segments
    # Count transitions from 0->1 and 1->0 along Y axis
    diff = torch.abs(binary[:, :, :, 1:, :] - binary[:, :, :, :-1, :])

    # Too many transitions = fragmented structure
    # Penalize excessive transitions
    transitions = diff.sum() / (binary.sum() + 1e-6)

    return transitions


def vae_loss(
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.001,
        physics_weight: float = 0.5,
        connectivity_weight: float = 0.3,
        ground_weight: float = 0.2,
        symmetry_weight: float = 0.1,
        vertical_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Calculate VAE loss with architectural constraints.

    Args:
        reconstruction: Reconstructed output [B, 1, 80, 42, 80].
        x: Original input [B, 1, 80, 42, 80].
        mu: Latent mean [B, latent_dim].
        logvar: Latent log variance [B, latent_dim].
        kl_weight: Weight for KL divergence term.
        physics_weight: Weight for physics constraint.
        connectivity_weight: Weight for connectivity constraint.
        ground_weight: Weight for ground plane constraint.
        symmetry_weight: Weight for symmetry encouragement.
        vertical_weight: Weight for vertical structure.

    Returns:
        Dictionary with all loss components.
    """
    # Binary cross-entropy for reconstruction
    bce_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    bce_loss = bce_loss / x.size(0)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)

    # Architectural constraints
    phys_loss = physics_loss(reconstruction)
    conn_loss = connectivity_loss(reconstruction)
    gnd_loss = ground_plane_loss(reconstruction)
    sym_loss = symmetry_loss(reconstruction)
    vert_loss = vertical_structure_loss(reconstruction)

    # Total loss
    total_loss = (
            bce_loss +
            kl_weight * kl_loss +
            physics_weight * phys_loss +
            connectivity_weight * conn_loss +
            ground_weight * gnd_loss +
            symmetry_weight * sym_loss +
            vertical_weight * vert_loss
    )

    return {
        'total_loss': total_loss,
        'reconstruction_loss': bce_loss,
        'kl_loss': kl_loss,
        'physics_loss': phys_loss,
        'connectivity_loss': conn_loss,
        'ground_loss': gnd_loss,
        'symmetry_loss': sym_loss,
        'vertical_loss': vert_loss,
    }


def main():
    """Test VAE architecture."""
    print("Testing 3D VAE with constraints...")

    # Create model
    model = VAE3D(latent_dim=256, input_shape=(80, 42, 80))

    # Test with random input (normalized to 0-1)
    batch_size = 4
    x = torch.rand(batch_size, 1, 80, 42, 80)  # Use rand instead of randn

    # Forward pass
    reconstruction, mu, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")

    # Test loss with constraints
    losses = vae_loss(reconstruction, x, mu, logvar)

    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✅ VAE architecture test passed!")


if __name__ == "__main__":
    main()
