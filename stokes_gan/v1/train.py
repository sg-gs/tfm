import os
import glob
from dataset import SimpleStokesDataset
from torch.utils.data import DataLoader
from StokesGAN import StokesGAN
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

# Configuración
config = {
    'base_filters': 32,
    'depth': 3,
    'disc_filters': 32,
    'disc_layers': 2,
    'lambda_adv': 1.0,
    'lambda_pixel': 10.0,
    'lambda_stokes': 5.0,
    'lr_g': 1e-4,
    'lr_d': 5e-5,
    'batch_size': 4,
    'num_epochs': 100
}

if torch.cuda.is_available():
    print('using CUDA')
    torch.backends.cudnn.benchmark = True
    

def load_ordered_hmi_files(data_directory):
    """
    Carga tus archivos hmi_training_*.fits en orden
    """
    pattern = os.path.join(data_directory, "hmi_training_*.fits")
    files = sorted(glob.glob(pattern))
    
    print(f"Encontrados {len(files)} archivos HMI ordenados")
    print(f"Primer archivo: {os.path.basename(files[0])}")
    print(f"Último archivo: {os.path.basename(files[-1])}")
    
    return files

def setup_training(data_directory):
    """
    Configura todo para entrenar tu StokesGAN
    """
    # Cargar archivos
    hmi_files = load_ordered_hmi_files(data_directory)
    
    # Dividir train/val (80/20)
    split_idx = int(0.8 * len(hmi_files))
    train_files = hmi_files[:split_idx]
    val_files = hmi_files[split_idx:]
    
    print(f"Train: {len(train_files)} archivos")
    print(f"Val: {len(val_files)} archivos")
    
    # Crear datasets
    train_dataset = SimpleStokesDataset(train_files, validation_mode=False)
    val_dataset = SimpleStokesDataset(val_files, validation_mode=True)

    can_use_gpu = torch.cuda.is_available()

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=(4 if can_use_gpu else 2),
        pin_memory=can_use_gpu,
        persistent_workers=can_use_gpu
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=(2 if can_use_gpu else 1),
        pin_memory=can_use_gpu,
        persistent_workers=can_use_gpu
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

def train(data_directory):
    """
    Entrenar tu StokesGAN con sistema de progreso avanzado
    """
    print("🚀 Configurando entrenamiento...")
    
    # Crear DataLoaders
    train_loader, val_loader = setup_training(data_directory)
    
    # Inicializar tu StokesGAN
    model = StokesGAN(config)
    
    # Tracking de métricas
    training_history = {
        'epoch': [],
        'g_loss': [],
        'd_loss': [],
        'g_adv_loss': [],
        'd_real_loss': [],
        'd_fake_loss': [],
        'val_loss': [],
        'learning_rate_g': [],
        'learning_rate_d': [],
        'time_per_epoch': []
    }
    
    # Configuración de progreso
    num_epochs = config['num_epochs']
    print(f"📊 Dataset: {len(train_loader)} batches de entrenamiento, {len(val_loader)} de validación")
    print(f"⏱️  Épocas totales: {num_epochs}")
    print(f"🎯 Dispositivo: {model.device}")
    
    # Crear carpeta para checkpoints
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"💾 Checkpoints en: {checkpoint_dir}")
    
    print("\n" + "="*80)
    print("🏋️ INICIANDO ENTRENAMIENTO")
    print("="*80)
    
    # Loop principal de entrenamiento
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Configurar modelos en modo entrenamiento
        model.generator.train()
        model.discriminator.train()
        
        # Métricas de la época
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        # Barra de progreso para la época
        epoch_pbar = tqdm(
            train_loader, 
            desc=f"📈 Época {epoch+1:03d}/{num_epochs:03d}",
            leave=False,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Entrenamiento batch por batch
        for batch_idx, (lr_stokes, hr_stokes) in enumerate(epoch_pbar):
            # Mover datos al dispositivo
            lr_stokes = lr_stokes.to(model.device)
            hr_stokes = hr_stokes.to(model.device)
            
            # Entrenar
            losses = model.train_step(lr_stokes, hr_stokes)
            
            # Acumular métricas
            for key, value in losses.items():
                epoch_metrics[key] += value
            
            # Actualizar barra de progreso con métricas actuales
            current_g_loss = epoch_metrics['g_loss'] / (batch_idx + 1)
            current_d_loss = epoch_metrics['d_loss'] / (batch_idx + 1)
            
            epoch_pbar.set_postfix({
                'G_Loss': f'{current_g_loss:.4f}',
                'D_Loss': f'{current_d_loss:.4f}',
                'Batch': f'{batch_idx+1}/{num_batches}'
            })
        
        # Calcular promedios de la época
        avg_metrics = {key: value / num_batches for key, value in epoch_metrics.items()}
        
        # Validación (opcional, cada pocas épocas)
        val_loss = 0.0
        if (epoch + 1) % 5 == 0 and val_loader:
            val_loss = validate_model(model, val_loader)
        
        # Tiempo de época
        epoch_time = time.time() - epoch_start_time
        
        # Guardar métricas
        training_history['epoch'].append(epoch + 1)
        training_history['g_loss'].append(avg_metrics['g_loss'])
        training_history['d_loss'].append(avg_metrics['d_loss'])
        training_history['g_adv_loss'].append(avg_metrics.get('g_adv_loss', 0))
        training_history['d_real_loss'].append(avg_metrics.get('d_real_loss', 0))
        training_history['d_fake_loss'].append(avg_metrics.get('d_fake_loss', 0))
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate_g'].append(model.opt_g.param_groups[0]['lr'])
        training_history['learning_rate_d'].append(model.opt_d.param_groups[0]['lr'])
        training_history['time_per_epoch'].append(epoch_time)
        
        # Mostrar resumen de época
        print(f"\n📊 Época {epoch+1:03d}/{num_epochs:03d} completada en {epoch_time:.1f}s")
        print(f"   🔸 G_Loss: {avg_metrics['g_loss']:.6f}")
        print(f"   🔸 D_Loss: {avg_metrics['d_loss']:.6f}")
        if 'g_adv_loss' in avg_metrics:
            print(f"   🔸 G_Adv:  {avg_metrics['g_adv_loss']:.6f}")
        if val_loss > 0:
            print(f"   🔸 Val_Loss: {val_loss:.6f}")
        
        # Estimación de tiempo restante
        avg_time_per_epoch = np.mean(training_history['time_per_epoch'])
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_time_per_epoch * remaining_epochs
        eta_formatted = format_time(eta_seconds)
        print(f"   ⏱️  ETA: {eta_formatted}")
        
        # Guardar checkpoint cada N épocas
        save_interval = config.get('save_interval', 20)
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'stokes_gan_epoch_{epoch+1:03d}.pth')
            save_checkpoint(model, training_history, epoch, checkpoint_path)
            print(f"   💾 Checkpoint guardado: {checkpoint_path}")
        
        # Plotear progreso cada 10 épocas
        if (epoch + 1) % 10 == 0:
            plot_training_progress(training_history, checkpoint_dir)
            print(f"   📈 Gráficos actualizados")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("🎉 ENTRENAMIENTO COMPLETADO!")
    print("="*80)
    
    # Resumen final
    total_time = sum(training_history['time_per_epoch'])
    print(f"⏱️  Tiempo total: {format_time(total_time)}")
    print(f"📊 G_Loss final: {training_history['g_loss'][-1]:.6f}")
    print(f"📊 D_Loss final: {training_history['d_loss'][-1]:.6f}")
    
    # Guardar historial completo
    history_path = os.path.join(checkpoint_dir, 'training_history.pth')
    torch.save(training_history, history_path)
    print(f"💾 Historial guardado: {history_path}")
    
    # Gráfico final
    plot_training_progress(training_history, checkpoint_dir, final=True)
    print(f"📈 Gráficos finales guardados en: {checkpoint_dir}")

    # Guardar modelo final completo
    final_model_path = os.path.join(checkpoint_dir, 'stokes_gan_final.pth')
    torch.save({
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_g_state_dict': model.opt_g.state_dict(),
        'optimizer_d_state_dict': model.opt_d.state_dict(),
        'config': config,
        'final_epoch': num_epochs,
        'final_metrics': {
            'g_loss': training_history['g_loss'][-1],
            'd_loss': training_history['d_loss'][-1]
        }
    }, final_model_path)
    print(f"🏆 Modelo final guardado: {final_model_path}")
    
    return model, training_history

def validate_model(model, val_loader):
    """Validación del modelo"""
    model.generator.eval()
    model.discriminator.eval()
    
    total_val_loss = 0.0
    num_val_batches = min(len(val_loader), 10)  # Validar solo algunos batches
    
    with torch.no_grad():
        for i, (lr_stokes, hr_stokes) in enumerate(val_loader):
            if i >= num_val_batches:
                break
                
            lr_stokes = lr_stokes.to(model.device)
            hr_stokes = hr_stokes.to(model.device)
            
            # Solo calcular pérdida del generador para validación
            fake_hr = model.generator(lr_stokes)
            g_loss = torch.nn.functional.mse_loss(fake_hr, hr_stokes)
            total_val_loss += g_loss.item()
    
    return total_val_loss / num_val_batches

def save_checkpoint(model, training_history, epoch, path):
    """Guardar checkpoint completo"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'opt_g_state_dict': model.opt_g.state_dict(),
        'opt_d_state_dict': model.opt_d.state_dict(),
        'config': config,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)

def plot_training_progress(history, save_dir, final=False):
    """Plotear progreso del entrenamiento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Progreso del Entrenamiento - StokesGAN', fontsize=16)
    
    epochs = history['epoch']
    
    # 1. Pérdidas principales
    axes[0, 0].plot(epochs, history['g_loss'], label='Generator Loss', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, history['d_loss'], label='Discriminator Loss', color='red', linewidth=2)
    if any(history['val_loss']):
        val_epochs = [e for e, v in zip(epochs, history['val_loss']) if v > 0]
        val_losses = [v for v in history['val_loss'] if v > 0]
        axes[0, 0].plot(val_epochs, val_losses, label='Validation Loss', color='green', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Pérdidas Principales')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pérdidas detalladas del discriminador
    if any(history['d_real_loss']) and any(history['d_fake_loss']):
        axes[0, 1].plot(epochs, history['d_real_loss'], label='D Real Loss', color='darkred', linewidth=2)
        axes[0, 1].plot(epochs, history['d_fake_loss'], label='D Fake Loss', color='orange', linewidth=2)
        axes[0, 1].set_title('Pérdidas del Discriminador')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning rates
    axes[1, 0].plot(epochs, history['learning_rate_g'], label='Generator LR', color='blue', linewidth=2)
    axes[1, 0].plot(epochs, history['learning_rate_d'], label='Discriminator LR', color='red', linewidth=2)
    axes[1, 0].set_title('Learning Rates')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. Tiempo por época
    axes[1, 1].plot(epochs, history['time_per_epoch'], color='purple', linewidth=2)
    axes[1, 1].set_title('Tiempo por Época')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Tiempo (segundos)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    prefix = 'final_' if final else 'progress_'
    plot_path = os.path.join(save_dir, f'{prefix}training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def format_time(seconds):
    """Formatear tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

# Función auxiliar para monitorear recursos
def print_system_info():
    """Mostrar información del sistema"""
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Memoria libre: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    else:
        print("💻 Usando CPU")

# Llamar antes del entrenamiento
print_system_info()
data_directory = '../../dataset_entrenamiento/'
trained_model = train(data_directory)
