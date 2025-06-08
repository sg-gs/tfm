import torch.nn as nn
import torch

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(8, 16, 4, stride=2, padding=1),  # Reducir de 32→16
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Añadir dropout
            
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # Reducir de 64→32
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # Reducir de 128→64
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, noisy, clean):
        x = torch.cat([noisy, clean], 1)
        return torch.sigmoid(self.model(x))
