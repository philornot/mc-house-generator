"""
3D Variational Autoencoder for Minecraft houses.
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

    def __init__(
            self,
            latent_dim: int = 256,
            input_shape: Tuple[int, int, int] = (80, 42, 80),
    ):
        """Initialize encoder.

        Args:
            latent_dim: Dimension of latent space (both mu and logvar).
            input_shape: Expected input shape (X, Y, Z).
        """
        super().__init__()

        self.latent_dim = latent_dim

        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        self.flatten_size = self._get_flatten_size(input_shape)

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate the flattened size after conv layers.

        Args:
            input_shape: Input shape (X, Y, Z).

        Returns:
            Flattened size after all conv layers.
        """
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

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder3D(nn.Module):
    """3D Convolutional Decoder for voxel data.

    Reconstructs 3D voxel grid from latent representation.
    Works with any output size.
    """

    def __init__(
            self,
            latent_dim: int = 256,
            output_shape: Tuple[int, int, int] = (80, 42, 80),
    ):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
            output_shape: Target output shape (X, Y, Z).
        """
        super().__init__()

        self.output_shape = output_shape

        self.feature_x = max(1, output_shape[0] // 16)
        self.feature_y = max(1, output_shape[1] // 16)
        self.feature_z = max(1, output_shape[2] // 16)

        self.flatten_size = 256 * self.feature_x * self.feature_y * self.feature_z

        self.fc = nn.Linear(latent_dim, self.flatten_size)

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
            Reconstructed voxel tensor [B, 1, X, Y, Z] in range [0, 1].
        """
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.feature_x, self.feature_y, self.feature_z)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)

        # Sigmoid BEFORE any spatial adjustments - so output is always [0, 1]
        x = torch.sigmoid(x)

        # Adjust to exact target shape if needed (crop or pad)
        current_shape = x.shape[2:]
        target_shape = self.output_shape

        for dim_idx, (current, target) in enumerate(zip(current_shape, target_shape)):
            if current > target:
                slices = [slice(None)] * x.ndim
                slices[dim_idx + 2] = slice(0, target)
                x = x[slices]
            elif current < target:
                pad_amount = target - current
                # F.pad order: last dim first → (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)
                pad = [0, 0, 0, 0, 0, 0]
                pad_index = 2 * (2 - dim_idx)  # maps dim 0→4, 1→2, 2→0
                pad[pad_index + 1] = pad_amount
                x = F.pad(x, pad, value=0.0)

        return x


class VAE3D(nn.Module):
    """3D Variational Autoencoder with architectural constraints."""

    # FIX: clamp logvar so std never blows up → prevents KL explosion
    LOGVAR_MIN = -10.0
    LOGVAR_MAX = 2.0

    def __init__(
            self,
            latent_dim: int = 256,
            input_shape: Tuple[int, int, int] = (80, 42, 80),
    ):
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
        """Reparameterization trick for sampling.

        Clamps logvar to prevent posterior collapse caused by exploding KL.

        Args:
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log variance of latent distribution [B, latent_dim].

        Returns:
            Sampled latent vector [B, latent_dim].
        """
        logvar = torch.clamp(logvar, self.LOGVAR_MIN, self.LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input voxel tensor [B, 1, X, Y, Z].

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, self.LOGVAR_MIN, self.LOGVAR_MAX)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples from the prior N(0, I).

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated voxel tensors [num_samples, 1, X, Y, Z].
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (returns mean, no sampling).

        Args:
            x: Input voxel tensor [B, 1, X, Y, Z].

        Returns:
            Latent mean vector [B, latent_dim].
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to voxel space.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed voxel tensor [B, 1, X, Y, Z].
        """
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Architectural constraint losses
# ---------------------------------------------------------------------------

def physics_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalize floating blocks (no support directly below).

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] with values in [0, 1].
        threshold: Value above which a voxel is considered solid.

    Returns:
        Scalar physics loss.
    """
    binary = (voxels > threshold).float()
    blocks_above_ground = binary[:, :, :, 1:, :]
    blocks_below = binary[:, :, :, :-1, :]
    floating = blocks_above_ground * (1.0 - blocks_below)
    return floating.mean()


def connectivity_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalize isolated blocks (no 6-connected neighbours).

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] with values in [0, 1].
        threshold: Value above which a voxel is considered solid.

    Returns:
        Scalar connectivity loss.
    """
    binary = (voxels > threshold).float()

    neighbors = torch.zeros_like(binary)
    neighbors[:, :, :-1, :, :] += binary[:, :, 1:, :, :]
    neighbors[:, :, 1:, :, :] += binary[:, :, :-1, :, :]
    neighbors[:, :, :, :-1, :] += binary[:, :, :, 1:, :]
    neighbors[:, :, :, 1:, :] += binary[:, :, :, :-1, :]
    neighbors[:, :, :, :, :-1] += binary[:, :, :, :, 1:]
    neighbors[:, :, :, :, 1:] += binary[:, :, :, :, :-1]

    isolated = binary * (neighbors == 0).float()

    if binary.sum() > 10:
        return isolated.mean()
    return torch.tensor(0.0, device=voxels.device)


def ground_plane_loss(voxels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Encourage blocks at ground level (Y=0).

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] with values in [0, 1].
        threshold: Value above which a voxel is considered solid.

    Returns:
        Scalar ground-plane loss.
    """
    binary = (voxels > threshold).float()
    ground_blocks = binary[:, :, :, 0, :].sum()
    total_blocks = binary.sum()

    if total_blocks > 0:
        ground_ratio = ground_blocks / total_blocks
        target_ratio = 0.05
        return F.relu(target_ratio - ground_ratio)

    return torch.tensor(0.0, device=voxels.device)


def symmetry_loss(voxels: torch.Tensor) -> torch.Tensor:
    """Encourage symmetry along the Z axis.

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] with values in [0, 1].

    Returns:
        Scalar symmetry loss.
    """
    flipped = torch.flip(voxels, dims=[4])
    return F.mse_loss(voxels, flipped)


def vertical_structure_loss(
        voxels: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Penalize excessive vertical fragmentation (too many 0↔1 transitions).

    Args:
        voxels: Voxel tensor [B, 1, X, Y, Z] with values in [0, 1].
        threshold: Value above which a voxel is considered solid.

    Returns:
        Scalar vertical-structure loss.
    """
    binary = (voxels > threshold).float()
    diff = torch.abs(binary[:, :, :, 1:, :] - binary[:, :, :, :-1, :])
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
    """Calculate total VAE loss with architectural constraints.

    Args:
        reconstruction: Reconstructed output [B, 1, X, Y, Z].
        x: Original input [B, 1, X, Y, Z].
        mu: Latent mean [B, latent_dim].
        logvar: Latent log-variance [B, latent_dim] (already clamped by forward()).
        kl_weight: Scalar weight for KL term (use annealing in trainer).
        physics_weight: Weight for physics constraint.
        connectivity_weight: Weight for connectivity constraint.
        ground_weight: Weight for ground-plane constraint.
        symmetry_weight: Weight for symmetry encouragement.
        vertical_weight: Weight for vertical-structure constraint.

    Returns:
        Dictionary with keys: total_loss, reconstruction_loss, kl_loss,
        physics_loss, connectivity_loss, ground_loss, symmetry_loss, vertical_loss.
    """
    bce_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum') / x.size(0)

    # Clamp logvar again defensively before computing KL
    logvar_clamped = torch.clamp(logvar, VAE3D.LOGVAR_MIN, VAE3D.LOGVAR_MAX)
    kl_loss = -0.5 * torch.sum(
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
    ) / x.size(0)

    phys_loss = physics_loss(reconstruction)
    conn_loss = connectivity_loss(reconstruction)
    gnd_loss = ground_plane_loss(reconstruction)
    sym_loss = symmetry_loss(reconstruction)
    vert_loss = vertical_structure_loss(reconstruction)

    total_loss = (
            bce_loss
            + kl_weight * kl_loss
            + physics_weight * phys_loss
            + connectivity_weight * conn_loss
            + ground_weight * gnd_loss
            + symmetry_weight * sym_loss
            + vertical_weight * vert_loss
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
    """Quick smoke test of the VAE architecture."""
    print("Testing 3D VAE...")

    model = VAE3D(latent_dim=256, input_shape=(80, 42, 80))
    x = torch.rand(4, 1, 80, 42, 80)

    reconstruction, mu, logvar = model(x)
    print(f"Input:          {x.shape}")
    print(f"Reconstruction: {reconstruction.shape}")
    print(f"logvar range:   [{logvar.min():.2f}, {logvar.max():.2f}]")

    losses = vae_loss(reconstruction, x, mu, logvar)
    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("\n✅ VAE architecture test passed!")


if __name__ == "__main__":
    main()
