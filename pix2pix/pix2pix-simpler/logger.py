import csv
import os
import json
from datetime import datetime
import torch
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import numpy as np

class Logger:
    def __init__(self, experiment_name, save_dir="results"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.start_time = datetime.now()
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.training_file = os.path.join(save_dir, f"training_{experiment_name}_{timestamp}.csv")
        self.evaluation_file = os.path.join(save_dir, f"evaluation_{experiment_name}_{timestamp}.csv")
        self.config_file = os.path.join(save_dir, f"config_{experiment_name}_{timestamp}.json")
        
        self.training_headers = [
            'epoch', 'timestamp', 'g_loss', 'd_loss', 
            'adversarial_loss', 'l1_loss', 'lr_generator', 'lr_discriminator',
            'epoch_time_seconds', 'gpu_memory_mb', 'gpu_utilization_percent'
        ]
        
        self.evaluation_headers = [
            'epoch', 'timestamp', 
            'psnr_I', 'psnr_Q', 'psnr_U', 'psnr_V',
            'ssim_I', 'ssim_Q', 'ssim_U', 'ssim_V',
            'mse_I', 'mse_Q', 'mse_U', 'mse_V',
            'noise_reduction_I', 'noise_reduction_Q', 'noise_reduction_U', 'noise_reduction_V',
            'physics_compliance_rate', 'physics_violations_count',
            'mean_I', 'std_I', 'mean_Q', 'std_Q', 'mean_U', 'std_U', 'mean_V', 'std_V'
        ]
        
        self._init_csv_files()
    
        print(f"Store -> Training data will be stored in: {self.training_file}")
        print(f"Store -> Evaluation data will be stored in: {self.evaluation_file}")
        print(f"   - Config: {self.config_file}")
    
    def _init_csv_files(self):
        with open(self.training_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.training_headers)
        
        with open(self.evaluation_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.evaluation_headers)
    
    def save_config(self, config_dict, model_info=None):
        full_config = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'config': config_dict,
            'model_info': model_info or {},
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(full_config, f, indent=2, default=str)
    
    def log_epoch(self, epoch, losses, lr_g, lr_d, epoch_time):
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        
        row = [
            epoch,
            datetime.now().isoformat(),
            losses.get('g_loss', 0),
            losses.get('d_loss', 0),
            losses.get('adv_loss', 0),
            losses.get('l1_loss', 0),
            lr_g,
            lr_d,
            epoch_time,
            gpu_memory,
        ]
        
        with open(self.training_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_evaluation(self, epoch, original, denoised):
        metrics = {}
        stokes_names = ['I', 'Q', 'U', 'V']
        
        for i, name in enumerate(stokes_names):
            orig_channel = original[..., i]
            denoised_channel = denoised[..., i]
            
            # PSNR
            mse = np.mean((orig_channel - denoised_channel)**2)
            if mse > 0:
                psnr = 20 * np.log10(orig_channel.max() / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            try:
                ssim_val = ssim(orig_channel, denoised_channel, data_range=orig_channel.max()-orig_channel.min())
            except:
                ssim_val = 0.0
            
            orig_std = np.std(orig_channel)
            denoised_std = np.std(denoised_channel)
            noise_reduction = (orig_std - denoised_std) / orig_std if orig_std > 0 else 0
            
            metrics[name] = {
                'psnr': psnr,
                'ssim': ssim_val,
                'mse': mse,
                'noise_reduction': noise_reduction,
                'mean': np.mean(denoised_channel),
                'std': np.std(denoised_channel)
            }
        
        # Validación física
        I = denoised[..., 0]
        Q = denoised[..., 1]
        U = denoised[..., 2]
        V = denoised[..., 3]
        
        pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
        violations = np.sum(pol_intensity > I * 1.001)
        total_pixels = I.size
        compliance_rate = 1.0 - (violations / total_pixels)
        
        row = [
            epoch,
            datetime.now().isoformat(),
            metrics['I']['psnr'], metrics['Q']['psnr'], metrics['U']['psnr'], metrics['V']['psnr'],
            metrics['I']['ssim'], metrics['Q']['ssim'], metrics['U']['ssim'], metrics['V']['ssim'],
            metrics['I']['mse'], metrics['Q']['mse'], metrics['U']['mse'], metrics['V']['mse'],
            metrics['I']['noise_reduction'], metrics['Q']['noise_reduction'], 
            metrics['U']['noise_reduction'], metrics['V']['noise_reduction'],
            compliance_rate, violations,
            metrics['I']['mean'], metrics['I']['std'], metrics['Q']['mean'], metrics['Q']['std'],
            metrics['U']['mean'], metrics['U']['std'], metrics['V']['mean'], metrics['V']['std']
        ]
        
        with open(self.evaluation_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        return metrics, compliance_rate
    
    def generate_summary_report(self):
        summary_file = self.training_file.replace('training_', 'summary_').replace('.csv', '.txt')
        
        try:
            df_training = pd.read_csv(self.training_file)
            df_evaluation = pd.read_csv(self.evaluation_file)
            
            with open(summary_file, 'w') as f:
                f.write(f"=== RESUMEN ===\n")
                f.write(f"Experimento: {self.experiment_name}\n")
                f.write(f"Inicio: {self.start_time}\n")
                f.write(f"Fin: {datetime.now()}\n")
                f.write(f"Duración total: {datetime.now() - self.start_time}\n\n")
                
                f.write("=== ENTRENAMIENTO ===\n")
                f.write(f"Épocas totales: {len(df_training)}\n")
                f.write(f"G_Loss final: {df_training['g_loss'].iloc[-1]:.6f}\n")
                f.write(f"D_Loss final: {df_training['d_loss'].iloc[-1]:.6f}\n")
                f.write(f"Tiempo promedio por época: {df_training['epoch_time_seconds'].mean():.1f}s\n")
                f.write(f"Memoria GPU máxima: {df_training['gpu_memory_mb'].max():.1f} MB\n\n")
                
                if len(df_evaluation) > 0:
                    f.write("=== EVALUACIÓN FINAL ===\n")
                    final_eval = df_evaluation.iloc[-1]
                    f.write(f"PSNR I: {final_eval['psnr_I']:.1f} dB\n")
                    f.write(f"PSNR Q: {final_eval['psnr_Q']:.1f} dB\n")
                    f.write(f"PSNR U: {final_eval['psnr_U']:.1f} dB\n")
                    f.write(f"PSNR V: {final_eval['psnr_V']:.1f} dB\n")
                    f.write(f"Cumplimiento físico: {final_eval['physics_compliance_rate']:.4f}\n")
            
            print(f"📄 Resumen generado: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ Error generando resumen: {e}")