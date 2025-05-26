import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class StokesLoss(nn.Module):
    """
    Función de pérdida que incorpora restricciones físicas de los parámetros de Stokes
    """
    def __init__(
        self, 
        lambda_pixel=10.0,
        # 50% para restricciones físicas fundamentales (Stokes es inviolable en el mundo real)
        lambda_stokes=50.0,
        # 20% Q,U,V deben ser cercanos a 0: restricción específica del dominio solar
        lambda_polarization=20.0,
        # 15% preservación de estructuras magnéticas
        lambda_magnetic=15.0,
        lambda_poisson=5.0,
        lambda_spatial=2.0,
        lambda_adv=1.0 
    ):
        super(StokesLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_stokes = lambda_stokes
        self.lambda_polarization = lambda_polarization
        self.lambda_magnetic = lambda_magnetic
        self.lambda_poisson = lambda_poisson
        self.lambda_spatial = lambda_spatial
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def sun_polarization_loss(self, pred_stokes):
        pred_Q, pred_U, pred_V = pred_stokes[:, 1], pred_stokes[:, 2], pred_stokes[:, 3]
    
        # Umbral de 0.1% acorde a conocimiento experto
        threshold = 0.001
        
        # Pérdida cuando Q, U, V exceden el umbral esperado
        q_loss = F.relu(torch.abs(pred_Q) - threshold).mean()
        u_loss = F.relu(torch.abs(pred_U) - threshold).mean()
        v_loss = F.relu(torch.abs(pred_V) - threshold).mean()
        
        return q_loss + u_loss + v_loss
    
    def magnetic_field_preservation_loss(self, pred_stokes, target_stokes):
        """
            Preserva la señal en regiones de campo magnético fuerte

            'En las zonas con campos magnéticos fuertes, deberían preservarse
            estos campos después de quitar el ruido'
        """
        target_I = target_stokes[:, 0]
        pred_I = pred_stokes[:, 0]

        # 0.1% como máximo para Q, U, V según los astrónomos
        threshold_magnetic = 1e-3
        
        # Identificar regiones de campo fuerte basándose en gradientes de intensidad
        # o umbrales de polarización
        magnetic_mask = (torch.sqrt(target_stokes[:, 1]**2 + target_stokes[:, 2]**2 + target_stokes[:, 3]**2) > threshold_magnetic)
        
        # Pérdida más alta en regiones magnéticas
        weighted_loss = self.mse_loss(pred_I * magnetic_mask, target_I * magnetic_mask)
        
        return weighted_loss

    def poisson_statistics_loss(self, pred_stokes):
        """Verifica que la estadística siga distribución de Poisson"""
        pred_I = pred_stokes[:, 0]
        
        # En regiones homogéneas, varianza ≈ media (propiedad de Poisson)
        local_mean = F.avg_pool2d(pred_I, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d((pred_I - local_mean)**2, kernel_size=5, stride=1, padding=2)
        
        # Penalizar desviaciones de la relación Poissoniana
        poisson_violation = F.mse_loss(local_var, local_mean)
        
        return poisson_violation

    def spatial_coherence_loss(self, pred_stokes):
        """
            Penaliza discontinuidades espaciales bruscas

            'Píxeles tienen coherencia espacial'
        """
        pred_I = pred_stokes[:, 0]

        # Gradientes en x e y
        grad_x = torch.abs(pred_I[..., :, 1:] - pred_I[..., :, :-1])
        grad_y = torch.abs(pred_I[..., 1:, :] - pred_I[..., :-1, :])
        
        # Penalizar gradientes excesivos (preservando bordes naturales)
        smoothness_loss = grad_x.mean() + grad_y.mean()
        
        return smoothness_loss

    def stokes_consistency_loss(self, pred_stokes, target_stokes):
        """
        Versión estabilizada de la restricción física de Stokes
        """
        pred_I, pred_Q, pred_U, pred_V = pred_stokes[:, 0], pred_stokes[:, 1], pred_stokes[:, 2], pred_stokes[:, 3]
        target_I, target_Q, target_U, target_V = target_stokes[:, 0], target_stokes[:, 1], target_stokes[:, 2], target_stokes[:, 3]

        # Restricción física: I² ≥ Q² + U² + V²
        pred_polarization = torch.sqrt(pred_Q**2 + pred_U**2 + pred_V**2)
        target_polarization = torch.sqrt(target_Q**2 + target_U**2 + target_V**2)

        # 1. Pérdida de violación física (esta parte está bien)
        violation_loss = F.relu(pred_polarization - pred_I).mean()

        # 2. SOLUCIÓN: Estabilizar el cálculo del grado de polarización
        epsilon = 1e-3  # Epsilon más grande para estabilidad

        # Solo calcular donde I es significativo
        mask = (pred_I > epsilon) & (target_I > epsilon)

        if mask.sum() > 0:
            # Aplicar máscara para evitar divisiones problemáticas
            pred_I_safe = pred_I[mask]
            target_I_safe = target_I[mask]
            pred_pol_safe = pred_polarization[mask]
            target_pol_safe = target_polarization[mask]
            
            # Grado de polarización estabilizado
            pred_degree = pred_pol_safe / (pred_I_safe + epsilon)
            target_degree = target_pol_safe / (target_I_safe + epsilon)
            
            # Clamping adicional para evitar valores extremos
            pred_degree = torch.clamp(pred_degree, 0, 1)
            target_degree = torch.clamp(target_degree, 0, 1)
            
            degree_loss = self.mse_loss(pred_degree, target_degree)
        else:
            degree_loss = torch.tensor(0.0, device=pred_stokes.device)
    
        return violation_loss + degree_loss
    
    def forward(self, pred, target, discriminator_output=None):
        pixel_loss = self.l1_loss(pred, target)
        stokes_loss = self.stokes_consistency_loss(pred, target)
        sun_polarization_loss = self.sun_polarization_loss(pred)
        magnetic_loss = self.magnetic_field_preservation_loss(pred, target)
        poisson_loss = self.poisson_statistics_loss(pred)
        spatial_loss = self.spatial_coherence_loss(pred)
            
        total_loss = (
            self.lambda_pixel * pixel_loss + 
            self.lambda_stokes * stokes_loss +
            self.lambda_polarization * sun_polarization_loss +
            self.lambda_magnetic * magnetic_loss +
            self.lambda_poisson * poisson_loss +
            self.lambda_spatial * spatial_loss
        )
        
        if discriminator_output is not None:
            # Pérdida adversarial para el generador
            adv_loss = -torch.mean(discriminator_output)
            total_loss += self.lambda_adv * adv_loss
        
        return total_loss, {
            'pixel': pixel_loss.item(),
            'stokes': stokes_loss.item(),
            'polarization': sun_polarization_loss.item(),
            'magnetic': magnetic_loss.item(),
            'poisson': poisson_loss.item(),
            'spatial': spatial_loss.item(),
            'adversarial': adv_loss.item() if discriminator_output is not None else 0
        }
