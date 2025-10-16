import torch
import torch.nn as nn

class ConNet(nn.Module):
    '''Convolutional Neural Network for image processing with time conditioning.'''
    def __init__(self, in_channels=3, hidden_channels=100, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        
    def forward(self, x_input, t):
        batch_size, _, H, W = x_input.shape
        t_expanded = t.view(batch_size, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([x_input, t_expanded], dim=1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
    
class Rectifiedflow():
    def __init__(self,model=None, num_steps=100):
        
        self.model = model # Aquí pasas una instancia de ConvNet
        self.N = num_steps # Pasos de integración (Euler)
    
    def get_train_tuple(self, z0=None, z1=None):
        # z0, z1: [batch_size, 3, 64, 64]
        t = torch.rand((z1.shape[0],1),device=z1.device) # [batch_size, 1]
        # Interpolación entre z0 y z1 → [batch_size, 3, 64, 64]
        z_t = t[:, None, None, None] * z1 + (1. - t[:, None, None, None]) * z0
        target = z1 - z0  # Lo que debe aprender
        return z_t, t, target  # t sigue siendo [batch_size, 1]
    
    @torch.no_grad()    
    def sample_ode(self, z0=None, N=None):
        """
        z0: [batch_size, 3, 64, 64]
        N: número de pasos (si no se pasa, usa self.N)
        """
        if N is None:
            N = self.N

        dt = 1. / N
        traj = []
        z = z0.detach().clone()
        batch_size = z.shape[0]
        
        traj.append(z.clone())

        for i in range(N):
            t = torch.ones((batch_size, 1), device=z.device) * (i / N)
            pred = self.model(z, t)  # [batch_size, 3, 64, 64]
            z = z + pred * dt
            traj.append(z.clone())

        return traj  # lista de [batch_size, 3, 64, 64]