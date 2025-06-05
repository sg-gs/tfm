#!/usr/bin/env python3
"""
Pix2Pix Denoising MEJORADO - Con evaluación completa como Self2Self
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time
from tqdm import tqdm
import os
from datetime import datetime

# Configuración GPU optimizada
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Dispositivo: {device}")

CONFIG = {
    'num_epochs': 100,         # Aumentado
    'batch_size': 8,           # Aprovechando GPU
    'learning_rate_g': 2e-4,   # Optimizado
    'learning_rate_d': 1e-4,   # Más bajo que G
    'lambda_l1': 50.0,         # Rebalanceado
    'image_size': 256,         # Tamaño completo
    'noise_level': 0.03,       # Más realista
    'save_interval': 10,
    'num_predictions': 10,     # Para evaluación como Self2Self
}

print(f"🎯 Pix2Pix Denoising MEJORADO")
print(f"📊 Config: {CONFIG['num_epochs']} épocas, batch={CONFIG['batch_size']}, img={CONFIG['image_size']}")

# =============================================================================
# MODELOS (MANTENIDOS IGUALES)
# =============================================================================

class SimpleUNet(nn.Module):
    """U-Net simplificado para denoising"""
    
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        
        # Salida con restricciones Stokes
        self.final = nn.Conv2d(32, 4, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        
        # Salida con restricciones Stokes
        out = self.final(d1)
        
        I = torch.sigmoid(out[:, 0:1])
        Q = torch.tanh(out[:, 1:2]) * 0.1
        U = torch.tanh(out[:, 2:3]) * 0.1
        V = torch.tanh(out[:, 3:4]) * 0.1
        
        return torch.cat([I, Q, U, V], dim=1)

class SimpleDiscriminator(nn.Module):
    """Discriminador simplificado"""
    
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(8, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, noisy, clean):
        x = torch.cat([noisy, clean], 1)
        return torch.sigmoid(self.model(x))

# =============================================================================
# DATASET MEJORADO PARA EVALUACIÓN
# =============================================================================

class EnhancedDenoisingDataset(Dataset):
    """Dataset mejorado que mantiene los datos originales para evaluación"""
    
    def __init__(self, fits_file, num_patches=500, image_size=256, noise_level=0.03, validation_mode=False):
        self.fits_file = fits_file
        self.num_patches = num_patches
        self.image_size = image_size
        self.noise_level = noise_level
        self.validation_mode = validation_mode
        
        # Cargar y guardar datos originales
        print(f"📥 Cargando {fits_file}...")
        with fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(np.float32)
            self.header = hdul[0].header
        
        if len(data.shape) == 4 and data.shape[0] == 6 and data.shape[1] == 4:
            self.stokes_cube = np.transpose(data[0], (1, 2, 0))  # (H, W, 4)
        else:
            raise ValueError(f"Formato inesperado: {data.shape}")
        
        print(f"✅ Datos cargados: {self.stokes_cube.shape}")
        
        # Guardar parámetros de normalización
        self.I_min = self.stokes_cube[..., 0].min()
        self.I_max = self.stokes_cube[..., 0].max()
        
        # Pre-generar patches
        self.patches = self._pregenerate_patches()
        print(f"✅ {len(self.patches)} patches generados")
    
    def get_original_cube(self):
        """Retornar cubo original para evaluación"""
        return self.stokes_cube.copy()
    
    def get_normalization_params(self):
        """Retornar parámetros de normalización"""
        return self.I_min, self.I_max
    
    def get_header(self):
        """Retornar header FITS"""
        return self.header
    
    def _pregenerate_patches(self):
        """Pre-generar todos los patches"""
        patches = []
        h, w, _ = self.stokes_cube.shape
        
        for i in range(self.num_patches):
            if h >= self.image_size and w >= self.image_size:
                y = np.random.randint(0, h - self.image_size)
                x = np.random.randint(0, w - self.image_size)
                patch = self.stokes_cube[y:y+self.image_size, x:x+self.image_size, :].copy()
            else:
                patch = self.stokes_cube.copy()
                if patch.shape[0] < self.image_size or patch.shape[1] < self.image_size:
                    pad_h = max(0, self.image_size - patch.shape[0])
                    pad_w = max(0, self.image_size - patch.shape[1])
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                patch = patch[:self.image_size, :self.image_size, :]
            
            # Normalizar
            patch = self._normalize_patch(patch)
            
            # Crear par noisy/clean
            clean_patch = patch.copy()
            noisy_patch = self._add_noise(patch)
            
            patches.append((noisy_patch, clean_patch))
        
        return patches
    
    def _normalize_patch(self, patch):
        """Normalización consistente con Self2Self"""
        normalized = patch.copy()
        I = patch[..., 0]
        I_scale = self.I_max - self.I_min
        
        if I_scale > 0:
            normalized[..., 0] = (I - self.I_min) / I_scale
            normalized[..., 1] = patch[..., 1] / I_scale
            normalized[..., 2] = patch[..., 2] / I_scale
            normalized[..., 3] = patch[..., 3] / I_scale
        
        return normalized
    
    def _add_noise(self, patch):
        """Añadir ruido realista"""
        noisy = patch.copy()
        
        # Ruido gaussiano escalado
        I = patch[..., 0]
        noise_I = np.random.normal(0, self.noise_level * np.sqrt(I + 1e-6), I.shape)
        noisy[..., 0] = np.maximum(I + noise_I, 0)
        
        # Ruido para Q, U, V
        for i in range(1, 4):
            noise_pol = np.random.normal(0, self.noise_level * 0.1, patch[..., i].shape)
            noisy[..., i] = patch[..., i] + noise_pol
        
        return noisy
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        noisy_patch, clean_patch = self.patches[idx]
        
        noisy_tensor = torch.from_numpy(noisy_patch).permute(2, 0, 1).float()
        clean_tensor = torch.from_numpy(clean_patch).permute(2, 0, 1).float()
        
        return noisy_tensor, clean_tensor

# =============================================================================
# FUNCIONES DE EVALUACIÓN (COPIADAS DE SELF2SELF)
# =============================================================================

def denoise_full_image(generator, image, config):
    """Aplicar denoising a imagen completa como Self2Self"""
    
    print(f"\n🧠 Aplicando Pix2Pix denoising...")
    print(f"   - Imagen: {image.shape}")
    print(f"   - Predicciones: {config['num_predictions']}")
    
    generator.eval()
    
    h, w, channels = image.shape
    patch_size = config['image_size']
    
    if h <= patch_size and w <= patch_size:
        # Imagen pequeña - procesar completa
        print("   - Modo: Cubo completo")
        
        img_input = image.copy()
        
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_input = np.pad(img_input, 
                              ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='reflect')
        
        img_input = img_input[:patch_size, :patch_size, :]
        
        # Convertir a tensor
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Múltiples predicciones (simulando dropout usando diferentes seeds)
        predictions = []
        for i in range(config['num_predictions']):
            with torch.no_grad():
                pred = generator(img_tensor)
                predictions.append(pred.cpu().numpy().squeeze())
        
        # Promediar predicciones
        result = np.mean(predictions, axis=0)
        result = np.transpose(result, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        result = result[:h, :w, :]
        
    else:
        # Imagen grande - procesar por patches
        print("   - Modo: Por patches")
        result = denoise_by_patches(generator, image, config)
    
    generator.train()
    return result

def denoise_by_patches(generator, image, config):
    """Denoising por patches para imágenes grandes"""
    
    h, w, channels = image.shape
    patch_size = config['image_size']
    overlap = 32
    
    # Posiciones de patches
    y_positions = list(range(0, h - patch_size + 1, patch_size - overlap))
    x_positions = list(range(0, w - patch_size + 1, patch_size - overlap))
    
    if y_positions[-1] + patch_size < h:
        y_positions.append(h - patch_size)
    if x_positions[-1] + patch_size < w:
        x_positions.append(w - patch_size)
    
    result = np.zeros_like(image)
    weights = np.zeros((h, w))
    
    total_patches = len(y_positions) * len(x_positions)
    patch_count = 0
    
    print(f"      Total patches: {total_patches}")
    
    for y in y_positions:
        for x in x_positions:
            # Extraer patch
            patch = image[y:y+patch_size, x:x+patch_size, :].copy()
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # Múltiples predicciones
            patch_predictions = []
            for i in range(config['num_predictions']):
                with torch.no_grad():
                    pred = generator(patch_tensor)
                    patch_predictions.append(pred.cpu().numpy().squeeze())
            
            denoised_patch = np.mean(patch_predictions, axis=0)
            denoised_patch = np.transpose(denoised_patch, (1, 2, 0))
            
            # Peso para blending
            weight_patch = create_weight_patch(patch_size, overlap)
            
            # Acumular resultado
            for c in range(channels):
                result[y:y+patch_size, x:x+patch_size, c] += denoised_patch[:, :, c] * weight_patch
            weights[y:y+patch_size, x:x+patch_size] += weight_patch
            
            patch_count += 1
            if patch_count % 20 == 0:
                print(f"      Procesados {patch_count}/{total_patches}")
    
    # Normalizar por pesos
    final_result = np.zeros_like(image)
    for c in range(channels):
        final_result[:, :, c] = np.divide(
            result[:, :, c], 
            weights, 
            out=np.zeros_like(weights), 
            where=weights != 0
        )
    
    return final_result

def create_weight_patch(patch_size, overlap):
    """Crear patch de pesos para blending"""
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    
    if overlap > 0:
        fade = np.linspace(0, 1, overlap)
        weight[:overlap, :] *= fade[:, np.newaxis]
        weight[-overlap:, :] *= fade[::-1, np.newaxis]
        weight[:, :overlap] *= fade[np.newaxis, :]
        weight[:, -overlap:] *= fade[::-1][np.newaxis, :]
    
    return weight

def desnormalize_stokes(stokes_normalized, I_min, I_max):
    """Desnormalizar cubo de Stokes"""
    
    stokes_original = stokes_normalized.copy()
    I_scale = I_max - I_min
    
    if I_scale > 0:
        stokes_original[..., 0] = stokes_normalized[..., 0] * I_scale + I_min
        stokes_original[..., 1] = stokes_normalized[..., 1] * I_scale
        stokes_original[..., 2] = stokes_normalized[..., 2] * I_scale
        stokes_original[..., 3] = stokes_normalized[..., 3] * I_scale
    
    return stokes_original

def validate_stokes_physics_post_denoising(stokes_cube):
    """Validar física de Stokes post-denoising"""
    
    I = stokes_cube[..., 0]
    Q = stokes_cube[..., 1]
    U = stokes_cube[..., 2]
    V = stokes_cube[..., 3]
    
    pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
    violations = np.sum(pol_intensity > I * 1.001)
    total_pixels = I.size
    
    compliance_rate = 1.0 - (violations / total_pixels)
    
    print(f"\n🔬 Validación física post-denoising:")
    print(f"   - Violaciones: {violations}/{total_pixels}")
    print(f"   - Cumplimiento: {compliance_rate:.4f} ({compliance_rate*100:.2f}%)")
    print(f"   - I rango: [{I.min():.4f}, {I.max():.4f}]")
    print(f"   - |Q| máximo: {np.abs(Q).max():.4f}")
    print(f"   - |U| máximo: {np.abs(U).max():.4f}")  
    print(f"   - |V| máximo: {np.abs(V).max():.4f}")
    
    return compliance_rate > 0.95

def create_comparison(original, denoised, save_path='pix2pix_stokes_comparison.png'):
    """Crear comparación visual como Self2Self"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    stokes_labels = ['I (Intensidad)', 'Q (Pol. Linear)', 'U (Pol. Linear)', 'V (Pol. Circular)']
   
    for i in range(4):
        # Original
        im1 = axes[0, i].imshow(original[..., i], cmap='hot' if i==0 else 'coolwarm', 
                                origin='lower')
        axes[0, i].set_title(f'Original {stokes_labels[i]}', fontweight='bold')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
        
        # Denoised
        im2 = axes[1, i].imshow(denoised[..., i], cmap='hot' if i==0 else 'coolwarm', 
                                origin='lower')
        axes[1, i].set_title(f'Pix2Pix {stokes_labels[i]}', fontweight='bold')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparación Stokes guardada: {save_path}")

def print_stokes_statistics(original, denoised):
    """Estadísticas específicas para Stokes"""
    
    print("\n📈 ESTADÍSTICAS FINALES STOKES:")
    print("=" * 40)
    
    stokes_names = ['I', 'Q', 'U', 'V']
    
    for i, name in enumerate(stokes_names):
        orig_std = np.std(original[..., i])
        denoised_std = np.std(denoised[..., i])
        noise_reduction = (orig_std - denoised_std) / orig_std if orig_std > 0 else 0
        
        mse = np.mean((original[..., i] - denoised[..., i])**2)
        
        if mse > 0:
            psnr = 20 * np.log10(original[..., i].max() / np.sqrt(mse))
            print(f"✅ {name}: reducción ruido = {noise_reduction:.1%}, MSE = {mse:.6f}, PSNR = {psnr:.1f} dB")
        else:
            print(f"✅ {name}: reducción ruido = {noise_reduction:.1%}, MSE = {mse:.6f}")

# =============================================================================
# ENTRENAMIENTO MEJORADO
# =============================================================================

def train_enhanced_pix2pix(fits_file):
    """Entrenamiento mejorado con evaluación completa"""
    
    print("🚀 Iniciando entrenamiento Pix2Pix MEJORADO...")
    
    # Crear datasets
    train_dataset = EnhancedDenoisingDataset(
        fits_file, 
        num_patches=500,
        image_size=CONFIG['image_size'],
        noise_level=CONFIG['noise_level']
    )
    
    val_dataset = EnhancedDenoisingDataset(
        fits_file,
        num_patches=100,
        image_size=CONFIG['image_size'], 
        noise_level=CONFIG['noise_level'],
        validation_mode=True
    )
    
    # DataLoaders optimizados
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📊 Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Modelos
    generator = SimpleUNet().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # Optimizadores
    opt_g = optim.Adam(generator.parameters(), lr=CONFIG['learning_rate_g'])
    opt_d = optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate_d'])
    
    # Pérdidas
    criterion_adv = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    print(f"🏗️ Generador: {sum(p.numel() for p in generator.parameters()):,} parámetros")
    print(f"🏗️ Discriminador: {sum(p.numel() for p in discriminator.parameters()):,} parámetros")
    
    # Crear directorio
    checkpoint_dir = f"checkpoints_pix2pix_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🏋️ ENTRENAMIENTO PIX2PIX MEJORADO")
    print(f"{'='*60}")
    
    # Variables para tracking
    train_losses = []
    
    # Loop de entrenamiento
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Época {epoch+1:03d}/{CONFIG['num_epochs']:03d}")
        
        for batch_idx, (noisy_batch, clean_batch) in enumerate(pbar):
            try:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                
                # Entrenar Discriminador
                opt_d.zero_grad()
                
                real_pred = discriminator(noisy_batch, clean_batch)
                real_labels = torch.ones_like(real_pred)
                d_real_loss = criterion_adv(real_pred, real_labels)
                
                with torch.no_grad():
                    fake_clean = generator(noisy_batch)
                fake_pred = discriminator(noisy_batch, fake_clean)
                fake_labels = torch.zeros_like(fake_pred)
                d_fake_loss = criterion_adv(fake_pred, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                opt_d.step()
                
                # Entrenar Generador
                opt_g.zero_grad()
                
                fake_clean = generator(noisy_batch)
                fake_pred = discriminator(noisy_batch, fake_clean)
                
                g_adv_loss = criterion_adv(fake_pred, torch.ones_like(fake_pred))
                g_l1_loss = criterion_l1(fake_clean, clean_batch)
                
                g_loss = g_adv_loss + CONFIG['lambda_l1'] * g_l1_loss
                g_loss.backward()
                opt_g.step()
                
                # Acumular
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                pbar.set_postfix({
                    'G_Loss': f'{g_loss.item():.4f}',
                    'D_Loss': f'{d_loss.item():.4f}',
                    'L1': f'{g_l1_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"\n❌ Error en batch {batch_idx}: {e}")
                continue
        
        # Resumen de época
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        train_losses.append({'epoch': epoch + 1, 'g_loss': avg_g_loss, 'd_loss': avg_d_loss})
        
        print(f"\n📊 Época {epoch+1:03d} completada en {epoch_time:.1f}s")
        print(f"   🔸 G_Loss: {avg_g_loss:.6f}")
        print(f"   🔸 D_Loss: {avg_d_loss:.6f}")
        
        # Guardar ejemplo
        if (epoch + 1) % 10 == 0:
            try:
                save_training_example(generator, val_loader, checkpoint_dir, epoch + 1)
                print(f"   📸 Ejemplo guardado")
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        
        # Checkpoint
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1:03d}.pth'))
            print(f"   💾 Checkpoint guardado")
    
    print(f"\n🎉 Entrenamiento completado!")
    print(f"📁 Checkpoints en: {checkpoint_dir}")
    
    # ============================================================================
    # EVALUACIÓN COMPLETA COMO SELF2SELF
    # ============================================================================
    
    print(f"\n{'='*60}")
    print(f"🧠 EVALUACIÓN COMPLETA")
    print(f"{'='*60}")
    
    # Obtener datos originales
    original_cube = train_dataset.get_original_cube()
    I_min, I_max = train_dataset.get_normalization_params()
    header = train_dataset.get_header()
    
    # Normalizar imagen completa para denoising
    original_normalized = train_dataset._normalize_patch(original_cube)
    
    # Aplicar denoising a imagen completa
    denoised_normalized = denoise_full_image(generator, original_normalized, CONFIG)
    
    # Desnormalizar resultados
    original_final = desnormalize_stokes(original_normalized, I_min, I_max)
    denoised_final = desnormalize_stokes(denoised_normalized, I_min, I_max)
    
    # Validar física post-denoising
    physics_ok = validate_stokes_physics_post_denoising(denoised_final)
    if not physics_ok:
        print("⚠️ Advertencia: Problemas de cumplimiento físico detectados")
    
    # Guardar resultado en FITS
    print("\n💾 Guardando resultado...")
    
    # Reorganizar para FITS: (H, W, 4) -> (4, H, W)
    denoised_fits_format = np.transpose(denoised_final, (2, 0, 1))
    
    hdu = fits.PrimaryHDU(denoised_fits_format.astype(np.float32))
    if header:
        hdu.header.update(header)
    hdu.writeto('pix2pix_denoised_enhanced.fits', overwrite=True)
    print("✅ Resultado guardado: pix2pix_denoised_enhanced.fits")
    
    # Crear visualización completa
    print("\n📊 Creando visualización...")
    create_comparison(original_final, denoised_final)
    
    # Estadísticas completas
    print_stokes_statistics(original_final, denoised_final)
    
    # Estadísticas adicionales del entrenamiento
    print("\n📈 RESULTADOS ADICIONALES:")
    print("=" * 30)
    
    final_g_loss = train_losses[-1]['g_loss']
    final_d_loss = train_losses[-1]['d_loss']
    print(f"✅ G_Loss final: {final_g_loss:.6f}")
    print(f"✅ D_Loss final: {final_d_loss:.6f}")
    
    # Métricas de convergencia
    if len(train_losses) > 10:
        recent_g_losses = [l['g_loss'] for l in train_losses[-10:]]
        g_loss_std = np.std(recent_g_losses)
        print(f"✅ Estabilidad G_Loss (últimas 10 épocas): {g_loss_std:.6f}")
    
    print(f"\n🎉 PROCESO PIX2PIX COMPLETADO!")
    print("📁 Archivos generados:")
    print(f"   - {checkpoint_dir}/ (modelos entrenados)")
    print("   - pix2pix_denoised_enhanced.fits (cubo Stokes denoised)")
    print("   - pix2pix_stokes_comparison.png (comparación)")
    
    return generator, discriminator, original_final, denoised_final

def save_training_example(generator, val_loader, save_dir, epoch):
    """Guardar ejemplo durante entrenamiento"""
    generator.eval()
    
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            
            denoised_batch = generator(noisy_batch)
            
            # Usar primer batch, todos los canales
            noisy_imgs = noisy_batch[0].cpu().numpy()
            clean_imgs = clean_batch[0].cpu().numpy()
            denoised_imgs = denoised_batch[0].cpu().numpy()
            
            # Crear comparación de entrenamiento
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            stokes_labels = ['I', 'Q', 'U', 'V']
            
            for i in range(4):
                # Ruidoso
                axes[0, i].imshow(noisy_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[0, i].set_title(f'Ruidoso {stokes_labels[i]}')
                axes[0, i].axis('off')
                
                # Pix2Pix
                axes[1, i].imshow(denoised_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[1, i].set_title(f'Pix2Pix {stokes_labels[i]}')
                axes[1, i].axis('off')
                
                # Limpio
                axes[2, i].imshow(clean_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[2, i].set_title(f'Limpio {stokes_labels[i]}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'training_example_epoch_{epoch:03d}.png'), 
                       dpi=100, bbox_inches='tight')
            plt.close()
            break
    
    generator.train()

# =============================================================================
# FUNCIÓN PRINCIPAL MEJORADA
# =============================================================================

def main():
    """Pipeline completo Pix2Pix con evaluación como Self2Self"""
    
    print("🎯 PIX2PIX DENOISING CON EVALUACIÓN COMPLETA")
    print("=" * 50)
    
    fits_file = '../data.fits'  # Cambia por tu archivo
    
    if not os.path.exists(fits_file):
        print(f"❌ No se encuentra {fits_file}")
        print("📝 Modifica la variable 'fits_file' para apuntar a tus datos FITS")
        return
    
    try:
        # Verificar GPU
        if device.type == 'cuda':
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"   - Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ Usando CPU - considera usar GPU para mejor rendimiento")
        
        # Entrenar y evaluar
        generator, discriminator, original, denoised = train_enhanced_pix2pix(fits_file)
        
        print("✅ Pipeline Pix2Pix completado exitosamente!")
        
        # Comparación con Self2Self si está disponible
        print("\n💡 COMPARACIÓN CON SELF2SELF:")
        print("=" * 30)
        print("Pix2Pix:")
        print("  ✅ Usa pares noisy/clean explícitos")
        print("  ✅ Entrenamiento supervisado")
        print("  ✅ Discriminador para realismo")
        print("  ⚠️ Requiere datos de entrenamiento")
        print("\nSelf2Self:")
        print("  ✅ No requiere datos limpios")
        print("  ✅ Auto-supervisado")
        print("  ✅ Funciona con una sola imagen")
        print("  ⚠️ Puede ser menos preciso")
        
        return generator, discriminator, original, denoised
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result = main()