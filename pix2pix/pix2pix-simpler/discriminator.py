import torch 
import torch.nn as nn

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(8, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, noisy, clean):
        x = torch.cat([noisy, clean], 1)
        return torch.sigmoid(self.model(x))