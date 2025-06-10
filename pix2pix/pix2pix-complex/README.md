# Pix2Pix

Based on the paper by Isola (2018) et al and modified for solar denoising

## Generator

Using 4 channels as the input and as the output on the generator for each of the Stokes parameters (I,Q,U,V).

### Architecture

The generator follows the **U-Net architecture** from the pix2pix paper (Isola et al. 2018), specifically implementing the Section 6.1.1 architecture adapted for Stokes parameter processing:

**Encoder:** C64-C128-C256-C512-C512-C512-C512-C512  
**U-Net Decoder:** CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

Where:
- **Ck** = Conv-BatchNorm-ReLU with k filters
- **CDk** = Conv-BatchNorm-Dropout-ReLU with k filters (50% dropout)
- All filters are 4×4 with stride 2
- Skip connections between encoder layer i and decoder layer n-i

### Key Features

- **Skip Connections**: U-Net architecture with skip connections between mirrored encoder-decoder layers to preserve low-level information shared between input and output
- **Specialized Output Activation**: Custom final activation designed for Stokes parameter constraints:
  - **I parameter**: `sigmoid` activation (ensures positive values)
  - **Q, U, V parameters**: `tanh` activation scaled by 0.1 (allows negative values with controlled range)

### Input/Output

- **Input**: 4-channel tensor representing Stokes parameters (I, Q, U, V)
- **Output**: 4-channel tensor with physically constrained Stokes parameters
- **Image Resolution**: Supports variable input sizes through fully convolutional architecture

### Physical Constraints

The generator enforces physical constraints on Stokes parameters:
- **I**: Intensity must be positive (≥ 0)
- **Q, U, V**: Linear polarization components can be negative but are bounded to realistic ranges (-0.1 to 0.1)

## Discriminator

The discriminator implements a **PatchGAN** architecture that classifies whether each NxN patch in the image pair is real or fake, rather than classifying the entire image.

### Architecture

Following the pix2pix Section 6.1.2 specification with 3 layers (70×70 receptive field):

**Structure:** C64-C128-C256-C512

Where:
- **First layer**: Conv 4×4 stride 2, **no BatchNorm** (as recommended in the paper)
- **Intermediate layers**: Conv-BatchNorm-LeakyReLU (4×4, stride 2)
- **Penultimate layer**: Conv-BatchNorm-LeakyReLU (4×4, stride 1)
- **Final layer**: Conv → Sigmoid (4×4, stride 1, outputs single channel)

### Key Features

- **Conditional Input**: Receives concatenated input and target images (8 channels total)
  - Input image (noisy): 4 Stokes parameter channels
  - Target image (clean/generated): 4 Stokes parameter channels
- **PatchGAN Design**: Evaluates local image patches rather than global image structure
- **Receptive Field**: 70×70 patches (with n_layers=3)
- **Activation**: LeakyReLU with slope 0.2 throughout, Sigmoid output
- **Adaptive to Stokes data**: Handles 4-channel polarimetric input naturally
- **Feature scaling**: Channels double with each layer (capped at 8× base filters)

### Input Processing

```python
def forward(self, input_img, target_img):
    # Concatenate input and target as in conditional Pix2Pix
    x = torch.cat([input_img, target_img], 1)  # 8 channels
    return self.model(x)
```


## Loss function

The training employs a **multi-component loss function** that combines adversarial training with reconstruction objectives and physical constraints specific to Stokes parameters.

```python
g_loss = (
    CONFIG['lambda_adv'] * g_adv_loss +
    CONFIG['lambda_l1'] * g_l1_loss +
    CONFIG['lambda_stokes'] * stokes_loss
)
```

### Loss Components

### 1. Adversarial Loss
- **Type**: Binary Cross-Entropy Loss (`nn.BCELoss()`)
- **Purpose**: Encourages the generator to produce realistic images that fool the discriminator
- **Weight**: `lambda_adv` (controls adversarial training strength)

#### 2. L1 Reconstruction Loss
- **Type**: L1 Loss (`nn.L1Loss()`)
- **Purpose**: Pixel-wise reconstruction accuracy, reduces blurring compared to L2
- **Weight**: `lambda_l1` (balances reconstruction fidelity)

#### 3. Stokes-Specific Combined Loss
- **Type**: Custom multi-term loss function
- **Components**:
  ```python
  total_loss = mse_loss + 0.1 * physics_loss + 0.05 * smoothness_loss
  ```

### Stokes Physics Loss

Enforces the fundamental **Stokes parameter constraint**:

**√(Q² + U² + V²) ≤ I**

**Purpose**: Ensures physically valid polarization states by penalizing violations of the fundamental Stokes constraint.

### Spatial Smoothness Loss

Promotes **spatial coherence** across all Stokes parameters.

**Purpose**: Reduces noise and maintains spatial consistency by penalizing excessive gradients in all Stokes channels.

### Loss term weights

- **MSE Loss**: 1.0 (base reconstruction)
- **Physics Loss**: 0.1 (moderate physical constraint enforcement)
- **Smoothness Loss**: 0.05 (light spatial regularization)

### Advantages

1. **Physical Validity**: Ensures generated Stokes parameters respect polarization physics
2. **Multi-scale Optimization**: Combines pixel-level (L1), patch-level (adversarial), and constraint-level (physics) objectives
3. **Spatial Coherence**: Smoothness term reduces noise artifacts
4. **Balanced Training**: Weighted combination prevents any single loss from dominating

This comprehensive loss design is specifically tailored for Stokes parameter denoising, ensuring both visual quality and physical correctness of the polarimetric data.

Based on the training code, here are the three sections:

---

## Data Generation

The training data is generated using an **synthetic approach** that simulates realistic polarimetric observations:

- Creates training pairs from FITS files containing Stokes parameter cubes
- Train/Validation Split: 80/20 ratio with consistent random seeding
- Patch extraction: Random patches of configurable size (default 256×256) from larger images

### Normalization Strategy
- **I parameter**: Normalized to [0,1] using global min/max scaling
- **Q, U, V parameters**: Scaled relative to intensity range for physical consistency
- **Preservation**: Original normalization parameters stored for reconstruction

### Data Augmentation
- **Pre-generation**: All patches generated once and cached for consistent training
- **Random sampling**: Multiple random crops from source images
- **Padding**: Reflective padding for images smaller than patch size

---

## Training

The training follows a **two-stage adversarial optimization** with custom loss balancing for Stokes parameter constraints:

### Two-Phase Optimization
1. **Discriminator Training**:
   ```python
   # Real pairs (noisy input + clean target)
   real_pred = discriminator(noisy_batch, clean_batch)
   d_real_loss = criterion_adv(real_pred, torch.ones_like(real_pred))
   
   # Fake pairs (noisy input + generated output)
   fake_clean = generator(noisy_batch)
   fake_pred = discriminator(noisy_batch, fake_clean)
   d_fake_loss = criterion_adv(fake_pred, torch.zeros_like(fake_pred))
   ```

2. **Generator Training**:
   ```python
   g_loss = (
       CONFIG['lambda_adv'] * g_adv_loss +      # Fool discriminator
       CONFIG['lambda_l1'] * g_l1_loss +        # Pixel reconstruction
       CONFIG['lambda_stokes'] * stokes_loss    # Physical constraints
   )
   ```

### Learning Rate Scheduling
- **ReduceLROnPlateau**: Adaptive learning rate reduction
    - **Generator**: Factor 0.5, patience 20 epochs
    - **Discriminator**: Factor 0.7, patience 15 epochs (more aggressive)

---

## Validation

The validation process employs **full-image evaluation** with multiple prediction averaging:

### Denoising
```python
def denoise_full_image(generator, image, config):
    predictions = []
    for i in range(config['num_predictions']):
        with torch.no_grad():
            pred = generator(img_tensor)
            predictions.append(pred.cpu().numpy().squeeze())
    
    result = np.mean(predictions, axis=0)
```

### Patch-based processing
For large images exceeding patch size:
- **Overlapping patches**: 32-pixel overlap for smooth blending
- **Weighted averaging**: Cosine-weighted patch fusion

### Physical Validation
```python
def validate_stokes_physics_post_denoising(stokes_cube):
    I = stokes_cube[..., 0]
    Q, U, V = stokes_cube[..., 1], stokes_cube[..., 2], stokes_cube[..., 3]
    
    pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
    violations = np.sum(pol_intensity > I * 1.001)  # 0.1% tolerance
    compliance_rate = 1.0 - (violations / I.size)
```

### Metrics
- **Noise reduction**: Standard deviation comparison per Stokes parameter
- **PSNR Cclculation**: Peak Signal-to-Noise Ratio for each channel
- **Physics compliance**: Constraint violation percentage
- **Visual comparison**: Side-by-side plots of all Stokes parameters

### Output generation
- **FITS Export**: Formatted astronomical data with preserved headers
- **Logging**: Detailed experiment tracking with hyperparameters and results

The validation ensures both **quantitative accuracy** and **physical plausibility** of the denoised Stokes parameters, making the results suitable for scientific analysis.
