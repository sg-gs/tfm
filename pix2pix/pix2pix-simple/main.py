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
import sys

sys.path.append('..')
from generator import Pix2PixGenerator
from discriminator import Pix2PixDiscriminator
from logger import Logger
from loss import stokes_combined_loss 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = Logger("pix2pix_gan_simple_v1")
print(f"Device: {device}")

CONFIG = {
    'num_epochs': 150,          # Más épocas
    'batch_size': 16,           # DUPLICAR (tienes 23.7GB!)
    'image_size': 256,          # Mantener
    
    'learning_rate_g': 1e-4,    # REDUCIR de 2e-4
    'learning_rate_d': 5e-5,    # MUY BAJO para discriminador
    
    'lambda_l1': 200.0,         # AUMENTAR de 50 → 200
    'lambda_adv': 0.5,          # REDUCIR peso adversarial
    'lambda_stokes': 0.1,
    
    'noise_level': 0.02,        # REDUCIR de 0.03
    'num_predictions': 10,
    'save_interval': 15,
}

print(f"Pix2Pix Denoising")
print(f"Config: epochs={CONFIG['num_epochs']}, batch={CONFIG['batch_size']}, img={CONFIG['image_size']}")

class DenoisingDataset(Dataset):    
    def __init__(self, fits_file, num_patches=500, image_size=256, noise_level=0.03, 
                validation_mode=False, split='train', train_ratio=0.8):
        self.fits_file = fits_file
        self.num_patches = num_patches
        self.image_size = image_size
        self.noise_level = noise_level
        self.validation_mode = validation_mode
        self.split = split
        self.train_ratio = train_ratio
        
        print(f"Cargando {fits_file}...")
        with fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(np.float32)
            self.header = hdul[0].header
        
        if len(data.shape) == 4 and data.shape[0] == 6 and data.shape[1] == 4:
            self.stokes_cube = np.transpose(data[0], (1, 2, 0))  # (H, W, 4)
        else:
            raise ValueError(f"Formato inesperado: {data.shape}")
        
        print(f"Datos cargados: {self.stokes_cube.shape}")
        
        # Guardar parámetros de normalización
        self.I_min = self.stokes_cube[..., 0].min()
        self.I_max = self.stokes_cube[..., 0].max()
        
        # Pre-generar patches
        self.patches = self._pregenerate_patches()
        print(f"✅ {len(self.patches)} patches generados")
    
    def get_original_cube(self):
        return self.stokes_cube.copy()
    
    def get_normalization_params(self):
        return self.I_min, self.I_max
    
    def get_header(self):
        return self.header
    
    def _pregenerate_patches(self,random_seed=42):
        all_patches = []
        h, w, _ = self.stokes_cube.shape
        
        total_patches = int(self.num_patches / self.train_ratio) if self.split == 'train' else int(self.num_patches / (1 - self.train_ratio))
        
        for i in range(total_patches):
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
            
            patch = self._normalize_patch(patch)
            
            clean_patch = patch.copy()
            noisy_patch = self._add_noise(patch)
            
            all_patches.append((noisy_patch, clean_patch))
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(all_patches))
        
        n_train = int(len(all_patches) * self.train_ratio)
        
        if self.split == 'train':
            selected_indices = indices[:n_train]
        else:
            selected_indices = indices[n_train:]
        
        selected_patches = [all_patches[i] for i in selected_indices]
        
        return selected_patches[:self.num_patches]
    
    def _normalize_patch(self, patch):
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

def denoise_full_image(generator, image, config):
    generator.eval()
    
    h, w, channels = image.shape
    patch_size = config['image_size']
    
    if h <= patch_size and w <= patch_size:
        # Imagen pequeña - procesar completa
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
        result = denoise_by_patches(generator, image, config)
    
    generator.train()
    return result

# we take the approach of applying the denoising per patches for bigger images, in order 
# to try to reduce the noise
def denoise_by_patches(generator, image, config):
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

# Prevent cut-lining by using blending 
def create_weight_patch(patch_size, overlap):
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    
    if overlap > 0:
        fade = np.linspace(0, 1, overlap)
        weight[:overlap, :] *= fade[:, np.newaxis]
        weight[-overlap:, :] *= fade[::-1, np.newaxis]
        weight[:, :overlap] *= fade[np.newaxis, :]
        weight[:, -overlap:] *= fade[::-1][np.newaxis, :]
    
    return weight

def desnormalize_stokes(stokes_normalized, I_min, I_max):
    stokes_original = stokes_normalized.copy()
    I_scale = I_max - I_min
    
    if I_scale > 0:
        stokes_original[..., 0] = stokes_normalized[..., 0] * I_scale + I_min
        stokes_original[..., 1] = stokes_normalized[..., 1] * I_scale
        stokes_original[..., 2] = stokes_normalized[..., 2] * I_scale
        stokes_original[..., 3] = stokes_normalized[..., 3] * I_scale
    
    return stokes_original

def validate_stokes_physics_post_denoising(stokes_cube):
    I = stokes_cube[..., 0]
    Q = stokes_cube[..., 1]
    U = stokes_cube[..., 2]
    V = stokes_cube[..., 3]
    
    pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
    violations = np.sum(pol_intensity > I * 1.001)
    total_pixels = I.size
    
    compliance_rate = 1.0 - (violations / total_pixels)
    
    print(f"\nValidación física post-denoising:")
    print(f"   - Violaciones: {violations}/{total_pixels}")
    print(f"   - Cumplimiento: {compliance_rate:.4f} ({compliance_rate*100:.2f}%)")
    print(f"   - I rango: [{I.min():.4f}, {I.max():.4f}]")
    print(f"   - |Q| máximo: {np.abs(Q).max():.4f}")
    print(f"   - |U| máximo: {np.abs(U).max():.4f}")  
    print(f"   - |V| máximo: {np.abs(V).max():.4f}")
    
    return compliance_rate > 0.95

def create_comparison(original, denoised, save_path='pix2pix_stokes_comparison.png'):
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
    
    print(f"Comparación Stokes guardada: {save_path}")

def print_stokes_statistics(original, denoised):
    print("\nESTADÍSTICAS FINALES STOKES:")
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

def train(fits_file):
    train_dataset = DenoisingDataset(
        fits_file, 
        num_patches=500,
        split='train',
        train_ratio=0.8,
        image_size=CONFIG['image_size'],
        noise_level=CONFIG['noise_level']
    )
    
    val_dataset = DenoisingDataset(
        fits_file,
        num_patches=100,
        split='val',
        train_ratio=0.8,
        image_size=CONFIG['image_size'], 
        noise_level=CONFIG['noise_level'],
        validation_mode=True
    )
    
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
    
    generator = Pix2PixGenerator(ngf=8).to(device)
    discriminator = Pix2PixDiscriminator(ndf=16).to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=CONFIG['learning_rate_g'])
    opt_d = optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate_d'])

    logger.save_config(
        config_dict=CONFIG,
        model_info={
            'generator_params': sum(p.numel() for p in generator.parameters()),
            'discriminator_params': sum(p.numel() for p in discriminator.parameters()),
            'architecture': 'Pix2Pix (real)'
        }
    )

    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, mode='min', factor=0.5, patience=20,
    )
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
        opt_d, mode='min', factor=0.7, patience=15,
    )
    
    criterion_adv = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    print(f"> Generador: {sum(p.numel() for p in generator.parameters()):,} parámetros")
    print(f"> Discriminador: {sum(p.numel() for p in discriminator.parameters()):,} parámetros")
    
    checkpoint_dir = f"checkpoints_pix2pix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    
    train_losses = []
    
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_adv_loss = 0
        epoch_l1_loss = 0
        epoch_stokes_loss = 0
        
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
                stokes_loss = stokes_combined_loss(clean_batch, fake_clean)

                # stokes + l1 + adversarial
                g_loss = (
                    CONFIG['lambda_adv'] * g_adv_loss + 
                    CONFIG['lambda_l1'] * g_l1_loss + 
                    CONFIG['lambda_stokes'] * stokes_loss
                )

                g_loss.backward()
                opt_g.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_adv_loss += g_adv_loss.item()
                epoch_stokes_loss += stokes_loss.item()
                epoch_l1_loss += g_l1_loss.item()

                pbar.set_postfix({
                    'G_Loss': f'{g_loss.item():.4f}',
                    'D_Loss': f'{d_loss.item():.4f}',
                    'L1': f'{g_l1_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"\nError en batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_adv_loss = epoch_adv_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_stokes_loss = epoch_stokes_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        train_losses.append({
            'epoch': epoch + 1,
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'adv_loss': avg_adv_loss,
            'l1_loss': avg_l1_loss,
            'stokes_loss': avg_stokes_loss
        })
        
        logger.log_epoch(
            epoch=epoch+1,
            losses=train_losses[-1],
            lr_g=opt_g.param_groups[0]['lr'],
            lr_d=opt_d.param_groups[0]['lr'],
            epoch_time=epoch_time
        )

        print(f"\nÉpoca {epoch+1:03d} completada en {epoch_time:.1f}s")
        print(f"   🔸 G_Loss: {avg_g_loss:.6f}")
        print(f"   🔸 D_Loss: {avg_d_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            try:
                save_training_example(generator, val_loader, checkpoint_dir, epoch + 1)
                print(f"---- Ejemplo guardado")
            except Exception as e:
                print(f"---- Error: {e}")
        
        # Checkpoint
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1:03d}.pth'))
            print(f"Checkpoint guardado")
    
    print(f"\nTraining completed. Checkpoints located at: {checkpoint_dir}")

    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")
    
    # Prepare for denoising
    original_cube = train_dataset.get_original_cube()
    I_min, I_max = train_dataset.get_normalization_params()
    header = train_dataset.get_header()    
    original_normalized = train_dataset._normalize_patch(original_cube)
    
    # Denoise
    denoised_normalized = denoise_full_image(generator, original_normalized, CONFIG)
    
    # Denorm results
    original_final = desnormalize_stokes(original_normalized, I_min, I_max)
    denoised_final = desnormalize_stokes(denoised_normalized, I_min, I_max)

    logger.log_evaluation(CONFIG['num_epochs'], original_final, denoised_final)
    
    # Validate physics
    physics_ok = validate_stokes_physics_post_denoising(denoised_final)
    if not physics_ok:
        print("⚠️ Advertencia: Problemas de cumplimiento físico detectados")
    
    print_stokes_statistics(original_final, denoised_final)
    
    print("\nOTHER STATS:")
    print("=" * 30)
    
    final_g_loss = train_losses[-1]['g_loss']
    final_d_loss = train_losses[-1]['d_loss']
    print(f"G_Loss final: {final_g_loss:.6f}")
    print(f"D_Loss final: {final_d_loss:.6f}")
    
    # coverge stats
    if len(train_losses) > 10:
        recent_g_losses = [l['g_loss'] for l in train_losses[-10:]]
        g_loss_std = np.std(recent_g_losses)
        print(f"Estabilidad G_Loss (últimas 10 épocas): {g_loss_std:.6f}")
    
    return generator, discriminator, original_final, denoised_final

def save_training_example(generator, val_loader, save_dir, epoch):
    generator.eval()
    
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            
            denoised_batch = generator(noisy_batch)

            target_np = clean_batch[0].cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
            denoised_np = denoised_batch[0].cpu().numpy().transpose(1, 2, 0)

            metrics, compliance = logger.log_evaluation(epoch, target_np, denoised_np)
            
            # Usar primer batch, todos los canales
            noisy_imgs = noisy_batch[0].cpu().numpy()
            clean_imgs = clean_batch[0].cpu().numpy()
            denoised_imgs = denoised_batch[0].cpu().numpy()
            
            # Crear comparación de entrenamiento
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            stokes_labels = ['I', 'Q', 'U', 'V']
            
            for i in range(4):
                # Noisy
                axes[0, i].imshow(noisy_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[0, i].set_title(f'Noisy {stokes_labels[i]}')
                axes[0, i].axis('off')
                
                # Pix2Pix
                axes[1, i].imshow(denoised_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[1, i].set_title(f'Pix2Pix {stokes_labels[i]}')
                axes[1, i].axis('off')
                
                # Clean
                axes[2, i].imshow(clean_imgs[i], cmap='hot' if i==0 else 'coolwarm')
                axes[2, i].set_title(f'Clean {stokes_labels[i]}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'training_example_epoch_{epoch:03d}.png'), 
                       dpi=100, bbox_inches='tight')
            plt.close()
            break
    
    generator.train()

def main():    
    fits_file = '../data.fits'
    
    if not os.path.exists(fits_file):
        print(f"Cannot find {fits_file}")
        print("Check 'fits_file' variable value")
        return
    
    learning_rates_g = [5e-5, 1e-4, 2e-4]
    learning_rates_d = [2e-5, 5e-5, 1e-4]
    epochs_list = [100, 150, 200]  

    exp_count = 0
    total_experiments = len(learning_rates_g) * len(learning_rates_d) * len(epochs_list)

    try:
        if device.type == 'cuda':
            print(f"GPU detected: {torch.cuda.get_device_name()}")
            print(f"   - Availablte memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("WARNING ! Using CPU - consider using GPU for speeding up the training")
        
        for lr_g in learning_rates_g:
            for lr_d in learning_rates_d:
                for epochs in epochs_list:
                    exp_count += 1
                    start_time = time.time()

                    try:
                        CONFIG['learning_rate_g'] = lr_g
                        CONFIG['learning_rate_d'] = lr_d
                        CONFIG['num_epochs'] = epochs
                        print(f"Empieza entrenamiento {exp_count}/{total_experiments}, lr_g {lr_g}, lr_d {lr_d}, epochs {epochs}")

                        generator, discriminator, original, denoised = train(fits_file)
                        training_time = (time.time() - start_time) / 60

                        suffix = f"_exp{exp_count}_lrg{lr_g:.0e}_lrd{lr_d:.0e}_ep{epochs}"
                        
                        fits_name = f'pix2pix_gan_denoised{suffix}.fits'
                        denoised_fits_format = np.transpose(denoised, (2, 0, 1))
                        hdu = fits.PrimaryHDU(denoised_fits_format.astype(np.float32))
                        hdu.writeto(fits_name, overwrite=True)
                        print(f"FITS guardado: {fits_name}")
                        comparison_name = f'pix2pix_gan_comparison{suffix}.png'
                        create_comparison(original, denoised, comparison_name)

                        print(f"Training {exp_count}/{total_experiments} completed in {training_time:.1f} min")

                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
        
        print("Training completed.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result = main()