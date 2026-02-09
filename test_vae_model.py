"""Quick test to verify VAE model works correctly.

Run this before training to make sure everything is set up properly.
"""

import logging
import torch
from vae_model import VAE3D, vae_loss


def test_vae_model():
    """Test VAE model architecture and forward pass."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("VAE MODEL TEST")
    logger.info("=" * 60)

    # Test 1: Model creation
    logger.info("\n[TEST 1] Creating VAE model...")
    model = VAE3D(latent_dim=256)
    logger.info("✅ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Test 2: Forward pass
    logger.info("\n[TEST 2] Testing forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 1, 80, 42, 80)
    logger.info(f"Input shape: {x.shape}")

    reconstruction, mu, logvar = model(x)
    logger.info(f"Reconstruction shape: {reconstruction.shape}")
    logger.info(f"Mu shape: {mu.shape}")
    logger.info(f"Logvar shape: {logvar.shape}")

    assert reconstruction.shape == x.shape, "Reconstruction shape mismatch!"
    assert mu.shape == (batch_size, 256), "Mu shape mismatch!"
    assert logvar.shape == (batch_size, 256), "Logvar shape mismatch!"
    logger.info("✅ Forward pass successful")

    # Test 3: Loss calculation
    logger.info("\n[TEST 3] Testing loss calculation...")
    losses = vae_loss(reconstruction, x, mu, logvar, kl_weight=0.001)
    logger.info(f"Total loss: {losses['total_loss'].item():.4f}")
    logger.info(f"Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    logger.info(f"KL loss: {losses['kl_loss'].item():.4f}")
    logger.info("✅ Loss calculation successful")

    # Test 4: Sampling
    logger.info("\n[TEST 4] Testing sampling...")
    device = torch.device('cpu')
    model.eval()
    samples = model.sample(num_samples=3, device=device)
    logger.info(f"Generated samples shape: {samples.shape}")
    assert samples.shape == (3, 1, 80, 42, 80), "Sample shape mismatch!"
    logger.info(f"Sample value range: [{samples.min():.3f}, {samples.max():.3f}]")
    logger.info("✅ Sampling successful")

    # Test 5: Encoding and decoding
    logger.info("\n[TEST 5] Testing encode/decode...")
    z = model.encode(x)
    logger.info(f"Encoded shape: {z.shape}")

    decoded = model.decode(z)
    logger.info(f"Decoded shape: {decoded.shape}")
    assert decoded.shape == x.shape, "Decoded shape mismatch!"
    logger.info("✅ Encode/decode successful")

    # Test 6: GPU compatibility (if available)
    if torch.cuda.is_available():
        logger.info("\n[TEST 6] Testing GPU compatibility...")
        device = torch.device('cuda')
        model_gpu = VAE3D(latent_dim=256).to(device)
        x_gpu = torch.randn(2, 1, 80, 42, 80).to(device)

        reconstruction_gpu, mu_gpu, logvar_gpu = model_gpu(x_gpu)
        logger.info(f"GPU forward pass successful")
        logger.info(f"Device: {reconstruction_gpu.device}")
        logger.info("✅ GPU compatibility verified")
    else:
        logger.info("\n[TEST 6] CUDA not available, skipping GPU test")

    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED! ✅")
    logger.info("=" * 60)
    logger.info("\nYou're ready to train!")
    logger.info("Run: python train_vae.py --epochs 100 --batch_size 4")


if __name__ == "__main__":
    test_vae_model()