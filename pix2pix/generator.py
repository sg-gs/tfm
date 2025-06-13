import torch 
import torch.nn as nn

class Pix2PixGenerator(nn.Module):
    """
    Pix2Pix Isola et al 2018.
    Section 6.1.1 based + adaptation to 4 channels (I,Q,U,V Stokes parameters)
    and reduced to ngf=32 to make the comparison with Self2Self fairer
    
    Original Section 6.1.1 Architecture (scaled down by half):
    > Encoder: C32-C64-C128-C256-C256-C256-C256-C256
    > U-Net Decoder: CD256-CD512-CD512-C512-C512-C256-C128-C64
    
    Where:
    - Ck = Conv-BatchNorm-ReLU with k filters
    - CDk = Conv-BatchNorm-Dropout-ReLU with k filters (dropout 50%)
    - All convolutions are 4x4 spatial filters applied with stride 2
    - Skip connections between layer i and layer n-i
    """
    
    def __init__(self, input_nc=4, output_nc=4, ngf=32, use_dropout=True):
        super(Pix2PixGenerator, self).__init__()
        
        # ENCODER
        # C32 (sin BatchNorm en la primera capa según el paper)
        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        # C64
        self.down2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # C128
        self.down3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # C256
        self.down4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C256
        self.down5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C256
        self.down6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C256
        self.down7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C256 (bottleneck - sin BatchNorm para evitar zeroing con batch_size=1)
        self.down8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )
        
        # DECODER con skip connections
        # CD256 (desde bottleneck: 256 canales)
        self.up1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # CD512 (256 from up1 + 256 from skip d7 = 512 input channels)
        self.up2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # CD512 (256 from up2 + 256 from skip d6 = 512 input channels)
        self.up3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # C512 (256 from up3 + 256 from skip d5 = 512 input channels, sin dropout)
        self.up4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C512 (256 from up4 + 256 from skip d4 = 512 input channels)
        self.up5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # C256 (128 from up5 + 128 from skip d3 = 256 input channels)
        self.up6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # C128 (64 from up6 + 64 from skip d2 = 128 input channels)
        self.up7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        # C64 -> output_nc (32 from up7 + 32 from skip d1 = 64 input channels)
        self.up8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder
        u1 = self.up1(d8) 
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))
        
        # Use Tanh as original Pix2Pix
        output = torch.tanh(u8)  # [-1, 1]
        
        # Reescalar para restricciones físicas de Stokes
        I = (output[:, 0:1] + 1) / 2  # [-1,1] -> [0,1] (intensidad positiva)
        Q = output[:, 1:2] * 0.1      # [-1,1] -> [-0.1,0.1] (parámetro Stokes)
        U = output[:, 2:3] * 0.1      # [-1,1] -> [-0.1,0.1] (parámetro Stokes)
        V = output[:, 3:4] * 0.1      # [-1,1] -> [-0.1,0.1] (parámetro Stokes)
            
        return torch.cat([I, Q, U, V], dim=1)