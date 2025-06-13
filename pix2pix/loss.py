import torch
import torch.nn.functional as F

def stokes_combined_loss(y_true, y_pred):
    mse_loss = F.mse_loss(y_pred, y_true)
    
    # Pérdida por violación de restricciones físicas
    physics_loss = stokes_physics_loss(y_true, y_pred)
    
    # Pérdida de suavidad espacial para coherencia
    smoothness_loss = stokes_spatial_smoothness_loss(y_pred)
    
    total_loss = mse_loss + 0.1 * physics_loss + 0.05 * smoothness_loss

    return total_loss

def stokes_physics_loss(y_true, y_pred):
    """Pérdida por violación de restricciones físicas de Stokes"""
    I_pred = y_pred[:, 0, :, :]  # (batch, channel, height, width)
    Q_pred = y_pred[:, 1, :, :]
    U_pred = y_pred[:, 2, :, :]
    V_pred = y_pred[:, 3, :, :]
    
    # Restricción fundamental: sqrt(Q² + U² + V²) ≤ I
    pol_intensity = torch.sqrt(Q_pred**2 + U_pred**2 + V_pred**2 + 1e-8)
    constraint_violation = torch.clamp(pol_intensity - I_pred, min=0.0)
    
    # Penalizar violaciones
    physics_penalty = torch.mean(constraint_violation**2)
    return physics_penalty

def stokes_spatial_smoothness_loss(y_pred):
    """Pérdida de suavidad espacial para coherencia entre parámetros"""
    gradients_loss = 0.0
    
    for i in range(4):  # Para cada parámetro de Stokes
        channel = y_pred[:, i, :, :]  # (batch, height, width)
        
        # Gradientes horizontales y verticales
        grad_h = channel[:, :-1, :] - channel[:, 1:, :]
        grad_v = channel[:, :, :-1] - channel[:, :, 1:]
        
        # Penalizar gradientes excesivos
        gradients_loss += torch.mean(grad_h**2)
        gradients_loss += torch.mean(grad_v**2)
    
    return gradients_loss / 4.0  # Normalizar por número de canales
