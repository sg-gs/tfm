import torch.nn as nn
import torch

class Self2SelfGenerator(nn.Module):    
    def __init__(self, dropout_prob=0.3):
        super(Self2SelfGenerator, self).__init__()
        
        # ENCODER
        # EB1: 4 -> 48 channels, 256x256 -> 128x128
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # EB2: 48 -> 48 channels, 128x128 -> 64x64
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # EB3: 48 -> 96 channels, 64x64 -> 32x32
        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # EB4: 96 -> 96 channels, 32x32 -> 16x16
        self.enc4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # EB5: 96 -> 192 channels, 16x16 -> 8x8
        self.enc5 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2)
        
        # EB6: 192 -> 192 channels, 8x8 (sin pooling)
        self.enc6 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Dropout más intenso
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # DECODER
        # DB1: 8x8 -> 16x16 (192 + 192 -> 192)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # DB2: 16x16 -> 32x32 (192 + 96 -> 96)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192 + 96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # DB3: 32x32 -> 64x64 (96 + 48 -> 48)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # DB4: 64x64 -> 128x128 (48 + 48 -> 48)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # DB5: 128x128 -> 256x256 (48 + 48 -> 64)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec5 = nn.Sequential(
            nn.Conv2d(48 + 48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)  # Dropout suave al final
        )
        
        self.final = nn.Conv2d(32, 4, 1)
    
    def forward(self, x):
        # EB1: 256x256 -> 128x128
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        # EB2: 128x128 -> 64x64
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # EB3: 64x64 -> 32x32
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # EB4: 32x32 -> 16x16
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # EB5: 16x16 -> 8x8
        e5 = self.enc5(p4)
        p5 = self.pool5(e5)
        
        # EB6: 8x8 (no pooling)
        e6 = self.enc6(p5)
        
        # BOTTLENECK
        b = self.bottleneck(e6)
        
        # DB1: 8x8 -> 16x16 (concat con e5)
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e5], 1))
        
        # DB2: 16x16 -> 32x32 (concat con e4)
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e4], 1))
        
        # DB3: 32x32 -> 64x64 (concat con e3)
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        
        # DB4: 64x64 -> 128x128 (concat con e2)
        u4 = self.up4(d3)
        d4 = self.dec4(torch.cat([u4, e2], 1))
        
        # DB5: 128x128 -> 256x256 (concat con e1)
        u5 = self.up5(d4)
        d5 = self.dec5(torch.cat([u5, e1], 1))
        
        out = self.final(d5)
        
        # Aplicar restricciones físicas de Stokes
        I = torch.sigmoid(out[:, 0:1])
        Q = torch.tanh(out[:, 1:2]) * 0.1
        U = torch.tanh(out[:, 2:3]) * 0.1
        V = torch.tanh(out[:, 3:4]) * 0.1
        
        return torch.cat([I, Q, U, V], dim=1)
