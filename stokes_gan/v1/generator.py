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

class UpsampleBlock(nn.Module):
    """
    Bloque de upsampling seguro usando interpolación + Conv2D
    Más estable que ConvTranspose2d
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Upsampling con interpolación bilineal
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # Convolución para ajustar canales
        x = self.conv(x)
        return x

class StokesUNetGenerator(nn.Module):
    """
    Generador U-Net adaptado para parámetros de Stokes con upsampling seguro
    """
    def __init__(self, input_channels=4, output_channels=4, base_filters=32, depth=3):
        super(StokesUNetGenerator, self).__init__()
        self.depth = depth
        
        print(f"🏗️ Construyendo UNet SEGURO: base_filters={base_filters}, depth={depth}")
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Primera capa especial para procesar Stokes
        self.initial_conv = ConvBlock(input_channels, base_filters, use_bn=False)
        print(f"✅ Initial conv: {input_channels} -> {base_filters}")
        
        encoder_channels = [base_filters]  # [32]

        # Capas del encoder
        in_ch = base_filters
        for i in range(depth):
            out_ch = base_filters * (2 ** (i + 1))  # 64, 128, 256
            
            print(f"🔽 Encoder {i}: {in_ch} -> {out_ch}")
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(out_ch)
            in_ch = out_ch
        
        print(f"📊 Encoder channels: {encoder_channels}")
        
        # Bottleneck
        bottleneck_ch = in_ch * 2
        print(f"🔄 Bottleneck: {in_ch} -> {bottleneck_ch}")
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch)
        
        # Decoder (upsampling path) - USANDO INTERPOLACIÓN SEGURA
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_ch = bottleneck_ch

        skip_channels = encoder_channels[1:]  # [64, 128, 256]
        skip_channels.reverse()  # [256, 128, 64]
        
        print(f"📊 Skip channels (reversed): {skip_channels}")

        for i, skip_ch in enumerate(skip_channels):
            print(f"🔼 Decoder {i}:")
            print(f"   - Upsample: {in_ch} -> {skip_ch}")
            print(f"   - Concatenación: {skip_ch} + {skip_ch} = {skip_ch * 2}")
            print(f"   - ConvBlock: {skip_ch * 2} -> {skip_ch}")
            
            # CAMBIO CLAVE: Usar UpsampleBlock en lugar de ConvTranspose2d
            self.upsamples.append(UpsampleBlock(in_ch, skip_ch, scale_factor=2))
            
            # Después de concatenar: skip_ch + skip_ch = skip_ch * 2
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            
            in_ch = skip_ch
        
        # Capa final
        print(f"🎯 Final: {in_ch} -> {output_channels}")
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)
        print("✅ Arquitectura UNet SEGURA construida correctamente")

    def forward(self, x):
        # print(f"🚀 FORWARD START - Input: {x.shape}")
        
        # Verificar entrada
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        
        # Encoder path con skip connections
        skip_connections = []
        
        # print("🔄 Initial conv...")
        x = self.initial_conv(x)
        # print(f"✅ After initial conv: {x.shape}")
        skip_connections.append(x)
        
        # print("🔽 Encoder path...")
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            # print(f"  Encoder {i} input: {x.shape}")
            x = encoder(x)
            # print(f"  After encoder {i}: {x.shape}")
            skip_connections.append(x)
            
            x = pool(x)
            # print(f"  After pool {i}: {x.shape}")
        
        # print(f"📊 Skip connections shapes: {[s.shape for s in skip_connections]}")
        
        # Bottleneck
        # print(f"🔄 Bottleneck input: {x.shape}")
        x = self.bottleneck(x)
        # print(f"✅ After bottleneck: {x.shape}")
        
        # Decoder path
        # print("🔼 Decoder path...")
        decoder_skips = skip_connections[1:]  # Excluir initial_conv
        decoder_skips.reverse()  # [enc2, enc1, enc0]
        
        # print(f"📊 Decoder will use skips: {[s.shape for s in decoder_skips]}")
        
        for i, (decoder, upsample, skip) in enumerate(zip(self.decoders, self.upsamples, decoder_skips)):
            # print(f"  🔼 Decoder {i}:")
            # print(f"     Input to upsample: {x.shape}")
            # print(f"     Skip connection: {skip.shape}")
            
            try:
                # UPSAMPLING SEGURO con interpolación
                # print(f"     🔄 Aplicando upsampling seguro...")
                x_up = upsample(x)
                # print(f"     ✅ After safe upsample: {x_up.shape}")
                
                # Verificar que las dimensiones espaciales coincidan
                if x_up.shape[2:] != skip.shape[2:]:
                    # print(f"     ⚠️ Spatial mismatch: {x_up.shape[2:]} vs {skip.shape[2:]}")
                    x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    # print(f"     After interpolation fix: {x_up.shape}")
                
                # Concatenar
                # print(f"     🔄 Concatenating: {x_up.shape} + {skip.shape}")
                x = torch.cat([x_up, skip], dim=1)
                # print(f"     ✅ After concat: {x.shape}")
                
                # Decoder conv
                # print(f"     🔄 Aplicando decoder conv...")
                x = decoder(x)
                # print(f"     ✅ After decoder conv: {x.shape}")
                
            except Exception as e:
                # print(f"❌ ERROR in decoder {i}: {e}")
                # print(f"   Input shape: {x.shape}")
                # print(f"   Skip shape: {skip.shape}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Salida final
        # print(f"🎯 Final conv input: {x.shape}")
        x = self.final_conv(x)
        # print(f"✅ FORWARD COMPLETE - Output: {x.shape}")
        
        return x
