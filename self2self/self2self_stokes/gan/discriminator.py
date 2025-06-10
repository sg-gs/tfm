import torch 
import torch.nn as nn

class Pix2PixDiscriminator(nn.Module):
    """
    Pix2Pix Isola et al 2018. 
    Section 6.1.2 based + Stokes restrictions.
    """
    
    def __init__(self, input_nc=4, ndf=64, n_layers=3):
        super(Pix2PixDiscriminator, self).__init__()
        
        # El discriminador recibe input + output concatenados (8 canales total)
        # input_nc * 2 porque concatenamos imagen ruidosa + imagen limpia
        
        sequence = [
            # First layer: C64 (wout BatchNorm as the paper suggests)
            nn.Conv2d(input_nc * 2, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Intermediate layers: C128, C256, C512
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Penúltima capa
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Last layer: 1-channel mapping + Sigmoid
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input_img, target_img):
        """
        Args:
            input_img: imagen ruidosa (4 canales Stokes)
            target_img: imagen limpia/fake (4 canales Stokes)
        """
        # Concatenar input y target como en Pix2Pix condicional
        x = torch.cat([input_img, target_img], 1)  # 8 canales
        return self.model(x)
