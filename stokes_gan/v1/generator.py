import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class ConvBlock(nn.Module):
    """
    Bloque convolucional básico con normalización y activación
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, activation='leaky'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class StokesUNetGenerator(nn.Module):
    """
    Generador U-Net adaptado para parámetros de Stokes
    Configurable en profundidad y número de filtros

    - Entrada: los 4 parámetros de Stokes (input_channels=4)
    - Salida: los mismos 4 parámetros en alta resolución (output_channels=4)
    - Se define la reducción del pooling con el depth. P.ej: depth 3 = 64x64 se convierte en 8x8
    """
    def __init__(self, input_channels=4, output_channels=4, base_filters=64, depth=5):
        super(StokesUNetGenerator, self).__init__()
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Primera capa especial para procesar Stokes
        # Aprende a interpretar simultáneamente los cuatro parámetros de Stokes. 
        # Aprende a combinar la información de forma coherente de "cuatro fotografías
        # del mismo paisaje tomadas con filtros diferentes"
        self.initial_conv = ConvBlock(input_channels, base_filters, use_bn=False)
        
        encoder_channels = [base_filters]

        # Capas del encoder
        in_ch = base_filters
        for i in range(depth):
            out_ch = base_filters * (2 ** i)

            self.encoders.append(ConvBlock(in_ch, out_ch))

            # Reduce la dimensionalidad a la mitad
            self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(out_ch)
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = in_ch * 2
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch)
        
        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_ch = bottleneck_ch

        skip_channels = encoder_channels[1:]  # [64, 128, 256]
        skip_channels.reverse() 

        for i, skip_ch in enumerate(skip_channels):
            print(f'Decoder {i}:')
            print(f'  - Upsample: {in_ch} -> {skip_ch}')
            print(f'  - Concatenación: {skip_ch} + {skip_ch} = {skip_ch * 2}')
            print(f'  - ConvBlock: {skip_ch * 2} -> {skip_ch}')
            
            # Upsample para coincidir con el skip connection
            self.upsamples.append(nn.ConvTranspose2d(
                in_ch, skip_ch, 
                kernel_size=2, stride=2
            ))
            
            # Después de concatenar: skip_ch + skip_ch = skip_ch * 2
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            
            in_ch = skip_ch
        
        # Capa final: desde el último decoder hasta output_channels
        print(f'Final: {in_ch} -> {output_channels}')
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path con skip connections
        skip_connections = []
        x = self.initial_conv(x)
        skip_connections.append(x)
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        decoder_skips = skip_connections[1:]
        decoder_skips.reverse() 
        
        for decoder, upsample, skip in zip(self.decoders, self.upsamples, decoder_skips):
            x = upsample(x)
            # Concatenar skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Salida final
        x = self.final_conv(x)
        
        return x
