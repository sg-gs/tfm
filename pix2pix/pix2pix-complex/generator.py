import torch 
import torch.nn as nn

class Pix2PixGenerator(nn.Module):
    """
    Pix2Pix Isola et al 2018.
    Section 6.1.1 based + adaptation to 4 layers (I,Q,U,V Stokes parameters)
    
    Section 6.1.1 Arch:
    > Encoder: C64-C128-C256-C512-C512-C512-C512-C512
    > U-Net Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    
    Where:
    - Ck = Conv-BatchNorm-ReLU with k filters
    - CDk = Conv-BatchNorm-Dropout-ReLU with k filters (dropout 50%)
    - Filters 4x4, stride 2
    - Skip connections between layers i and n-i
    """
    
    def __init__(self, input_nc=4, output_nc=4, ngf=64, use_dropout=True):
        super(Pix2PixGenerator, self).__init__()
        
        # ENCODER
        # Primer layer: C64 (sin BatchNorm según el paper)
        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        # C128
        self.down2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # C256
        self.down3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # C512
        self.down4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C512
        self.down5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C512
        self.down6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C512
        self.down7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C512 (bottleneck)
        self.down8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
            # Sin BatchNorm en bottleneck para evitar zeroing con batch_size=1
        )
        
        # DECODER con skip connections
        # CD512 (desde bottleneck)
        self.up1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # CD1024 (512 + 512 de skip connection)
        self.up2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # CD1024 (512 + 512 de skip connection)
        self.up3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        # C1024 (512 + 512 de skip connection)
        self.up4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # C1024 (512 + 512 de skip connection)
        self.up5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # C512 (256 + 256 de skip connection)
        self.up6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # C256 (128 + 128 de skip connection)
        self.up7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        # C128 -> output_nc (64 + 64 de skip connection)
        self.up8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
        )
        
        # Activación final según el paper (Tanh)
        # Pero adaptada para restricciones Stokes
        self.final_activation = nn.Tanh()
    
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
        
        # I: debe ser positivo -> sigmoid
        I = torch.sigmoid(u8[:, 0:1])
        
        # Q, U, V: pueden ser negativos pero limitados -> tanh con escalado
        Q = torch.tanh(u8[:, 1:2]) * 0.1
        U = torch.tanh(u8[:, 2:3]) * 0.1  
        V = torch.tanh(u8[:, 3:4]) * 0.1
        
        return torch.cat([I, Q, U, V], dim=1)
