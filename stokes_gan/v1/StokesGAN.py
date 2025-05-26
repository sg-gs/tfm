import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from generator import StokesUNetGenerator
from discriminator import PatchGANDiscriminator
from loss_function import StokesLoss

class StokesGAN:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar redes
        self.generator = StokesUNetGenerator(
            input_channels=4,
            output_channels=4,
            base_filters=config.get('base_filters', 64),
            depth=config.get('depth', 4)
        ).to(self.device)
        
        self.discriminator = PatchGANDiscriminator(
            input_channels=8,
            base_filters=config.get('disc_filters', 64),
            num_layers=config.get('disc_layers', 3)
        ).to(self.device)
        
        # Función de pérdida
        self.criterion = StokesLoss(
            lambda_adv=config.get('lambda_adv', 1.0),
            lambda_pixel=config.get('lambda_pixel', 1.0),
            lambda_stokes=config.get('lambda_stokes', 0.1),
            lambda_polarization=config.get('lambda_polarization', 0.05),
            lambda_magnetic=config.get('lambda_magnetic', 0.02),
            lambda_poisson=config.get('lambda_poisson', 0.01),
            lambda_spatial=config.get('lambda_spatial', 0.005)
        )
        
        # Optimizadores
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.get('lr_g', 2e-4),
            betas=(0.5, 0.999)
        )
        
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('lr_d', 2e-4),
            betas=(0.5, 0.999)
        )
        
        # Criterio para discriminador
        self.adversarial_loss = nn.BCELoss()
    
    def train_step(self, lr_stokes, hr_stokes):
        """Un paso de entrenamiento"""
        batch_size = lr_stokes.size(0)
        
        # Labels para entrenamiento adversarial
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # ==================
        # Entrenar Discriminador
        # ==================
        self.opt_d.zero_grad()
        
        # Generar imágenes falsas
        with torch.no_grad():
            fake_hr = self.generator(lr_stokes)
        
        # Discriminador en imágenes reales
        real_pred = self.discriminator(lr_stokes, hr_stokes)
        real_pred_resized = F.adaptive_avg_pool2d(real_pred, (1, 1)).view(batch_size, -1)
        d_real_loss = self.adversarial_loss(real_pred_resized, real_label)
        
        # Discriminador en imágenes falsas
        fake_pred = self.discriminator(lr_stokes, fake_hr)
        fake_pred_resized = F.adaptive_avg_pool2d(fake_pred, (1, 1)).view(batch_size, -1)
        d_fake_loss = self.adversarial_loss(fake_pred_resized, fake_label)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        self.opt_d.step()
        
        # ==================
        # Entrenar Generador
        # ==================
        self.opt_g.zero_grad()
        
        # Generar imágenes
        fake_hr = self.generator(lr_stokes)
        
        # Pérdida del discriminador para el generador
        fake_pred = self.discriminator(lr_stokes, fake_hr)
        fake_pred_resized = F.adaptive_avg_pool2d(fake_pred, (1, 1)).view(batch_size, -1)
        
        # Pérdida total del generador
        g_loss, loss_dict = self.criterion(fake_hr, hr_stokes, fake_pred_resized)

        print(f"  📊 Desglose de pérdidas:")
        for key, value in loss_dict.items():
            print(f"     {key}: {value:.2e}")
        
        # Verificar gradientes
        max_grad = 0
        for param in self.generator.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        print(f"  🔥 Gradiente máximo: {max_grad:.2e}")

        print(f"  📊 Rangos de datos:")
        print(f"     pred_I: [{fake_hr[:, 0].min():.3f}, {fake_hr[:, 0].max():.3f}]")
        print(f"     target_I: [{hr_stokes[:, 0].min():.3f}, {hr_stokes[:, 0].max():.3f}]")

        g_loss.backward()
        self.opt_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            **loss_dict
        }
