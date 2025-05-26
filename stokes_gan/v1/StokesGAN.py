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
        """Versión gradual para integrar pérdidas físicas"""
        batch_size = lr_stokes.size(0)
        
        # ==================
        # Entrenar Discriminador
        # ==================
        self.opt_d.zero_grad()
        with torch.no_grad():
            fake_hr_for_d = self.generator(lr_stokes)
        
        real_pred = self.discriminator(lr_stokes, hr_stokes)
        fake_pred_for_d = self.discriminator(lr_stokes, fake_hr_for_d)
        
        d_loss = torch.mean((real_pred - 1)**2) + torch.mean(fake_pred_for_d**2)
        d_loss.backward()
        self.opt_d.step()
        
        # ==================
        # Entrenar Generador - AÑADIR PÉRDIDAS GRADUALMENTE
        # ==================
        self.opt_g.zero_grad()
        
        fake_hr = self.generator(lr_stokes)
        
        # PASO 1: Solo pérdida pixel (ya sabemos que funciona)
        pixel_loss = self.criterion.l1_loss(fake_hr, hr_stokes)
        
        # PASO 2: Añadir pérdida adversarial
        fake_pred = self.discriminator(lr_stokes, fake_hr)
        fake_pred_resized = F.adaptive_avg_pool2d(fake_pred, (1, 1)).view(batch_size, -1)
        adv_loss = torch.mean((fake_pred_resized - 1)**2)  # Simplificado
        
        # PASO 3: Añadir pérdidas físicas UNA POR UNA
        stokes_loss = self.criterion.stokes_consistency_loss(fake_hr, hr_stokes)
        polarization_loss = self.criterion.sun_polarization_loss(fake_hr)
        
        # PÉRDIDA TOTAL - Empezar con pesos muy pequeños
        total_loss = (
            1.0 * pixel_loss +           # Peso normal
            0.1 * adv_loss +             # Peso pequeño
            0.01 * stokes_loss +         # Peso muy pequeño
            0.01 * polarization_loss     # Peso muy pequeño
        )
        
        # DIAGNÓSTICO
        print(f"  📊 Pérdidas individuales:")
        print(f"     pixel: {pixel_loss.item():.6f}")
        print(f"     adversarial: {adv_loss.item():.6f}")
        print(f"     stokes: {stokes_loss.item():.6f}")
        print(f"     polarization: {polarization_loss.item():.6f}")
        print(f"     TOTAL: {total_loss.item():.6f}")
        
        total_loss.backward()
        
        # Verificar gradientes
        max_grad = 0
        for param in self.generator.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        print(f"  🔥 Gradiente máximo: {max_grad:.6f}")
        
        self.opt_g.step()
        
        return {
            'g_loss': total_loss.item(),
            'd_loss': d_loss.item(),
            'pixel': pixel_loss.item(),
            'stokes': stokes_loss.item(),
            'polarization': polarization_loss.item(),
            'adversarial': adv_loss.item(),
            'magnetic': 0,
            'poisson': 0,
            'spatial': 0
        }