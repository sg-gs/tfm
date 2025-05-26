import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from generator import ConvBlock

class PatchGANDiscriminator(nn.Module):
    """
    Discriminador PatchGAN configurable para evaluar autenticidad local
    """
    def __init__(self, input_channels=8, base_filters=64, num_layers=3, patch_size=70):
        super(PatchGANDiscriminator, self).__init__()
        
        # input_channels = 8 porque concatenamos input (4) + output (4) de Stokes
        
        layers = []
        
        # Primera capa sin normalización
        layers.append(ConvBlock(input_channels, base_filters, stride=2, use_bn=False))
        
        # Capas intermedias
        in_ch = base_filters
        for i in range(1, num_layers):
            out_ch = min(base_filters * (2 ** i), 512)  # Limitar a 512 filtros máximo
            layers.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        
        # Capa final sin stride para mantener resolución
        layers.append(ConvBlock(in_ch, in_ch * 2, stride=1))
        
        # Salida binaria (real/fake)
        layers.append(nn.Conv2d(in_ch * 2, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, input_image, target_image):
        # Concatenar imagen de entrada y objetivo
        x = torch.cat([input_image, target_image], dim=1)
        return torch.sigmoid(self.model(x))
