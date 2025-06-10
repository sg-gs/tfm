# Self2Self GAN

A GAN designed for solar denoising. The generator is based on the auto-denoising Self2Self
model of the paper by Quan (2020) et al and the discriminator is based on the PatchGAN of Pix2Pix
by Isola et al (2018) adapted for solar denoising

## Generator
- U-Net architecture with 6 encoder-decoder levels
- Skip connections to preserve details
- Extensive dropout in decoder for regularization
- Physical constraints in output for Stokes parameters

### Input/Output

- **Input**: 4-channel tensor representing Stokes parameters (I, Q, U, V)
- **Output**: 4-channel tensor with physically constrained Stokes parameters
- **Image Resolution**: Supports variable input sizes through fully convolutional architecture

### Physical Constraints

The generator enforces physical constraints on Stokes parameters:
- **I**: Intensity must be positive (≥ 0)
- **Q, U, V**: Sensibility of 0.1% of the variation

## Discriminator
- PatchGAN-type convolutional discriminator
- Receives 8 channels (noisy image + target concatenated)

### Loss Functions

The training employs a **multi-component loss function** that combines adversarial training with reconstruction objectives and physical constraints specific to Stokes parameters.

- **Self2Self**: Loss on pixels masked by Bernoulli sampling
- **Adversarial**: GAN loss for visual realism
- **L1**: Global pixel-to-pixel consistency
- **Stokes Physics**: Penalty for physical constraint violations

### Expected Data Structure
- FITS file with format: `(6, 4, H, W)` where first element `[0]` contains Stokes parameters
- Parameter order: `[I, Q, U, V]`

## Training Process

1. **Data Preparation**: 
   - Generate 128x128 pixel patches
   - Stokes-specific normalization
   - Bernoulli sampling with probability 0.3

2. **Alternating Training**:
   - Discriminator: Distinguishes between real and reconstructed images
   - Generator: Optimizes Self2Self + GAN loss combination

3. **Evaluation**: 
   - Full image denoising by patches
   - Ensemble of multiple predictions with dropout
   - Post-denoising physical constraint validation

## Physical Validation

The system automatically verifies that reconstructions maintain fundamental physical constraints:
- `sqrt(Q² + U² + V²) ≤ I` (polarized intensity doesn't exceed total intensity)
- Appropriate ranges for each Stokes parameter

## Reported Metrics

- **Noise reduction** per Stokes parameter
- **PSNR/MSE** for each component
- **Physical compliance** (percentage of valid pixels)
- **Training stability** (loss variance)

## Advantages over Pix2Pix

- **No clean data required**: Completely self-supervised learning
- **Works with single image**: No external datasets needed
- **Better generalization**: Image-specific training
- **Physical preservation**: Domain-specific astronomical constraints

## Limitations

- **Training time**: Requires individual training per image
- **Spatial redundancy dependence**: Limited effectiveness in regions with unique information
- **Loss balancing**: Requires careful adjustment of λ weights
