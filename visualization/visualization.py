import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import random
def sample_rectified_flow(model, Z0_batch, num_steps, device):
    """
    Simula la ODE del Rectified Flow utilizando el método de Euler.

    Args:
        model (nn.Module): El modelo U-Net entrenado v(X, t).
        Z0_batch (torch.Tensor): Lote de ruido inicial (Z0 ~ π0).
        num_steps (int): Número de pasos de discretización (N).
        device (torch.device): Dispositivo (CPU o CUDA).

    Returns:
        torch.Tensor: Lote de imágenes generadas (Z1).
    """
    # CRÍTICO: Poner el modelo en modo de evaluación y desactivar gradientes
    model.eval() 
    Z_t = Z0_batch.to(device)
    # Lista para almacenar las trayectorias: [Z0, Z_t1, Z_t2, ..., Z1]
    # Inicia con Z0
    trajectories = [Z_t.cpu().clone()]
    # El tiempo avanza de 0 a 1
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            # Tiempo actual t (Euler usa el punto inicial del intervalo)
            t_start = i * dt
            
            # Convertir el tiempo escalar a un tensor 1D (B) para el modelo
            t_tensor = torch.full((Z_t.size(0),), t_start, device=device)
            
            # 1. Obtener la velocidad predicha v(Zt, t)
            # Pasamos Z_t (imagen) y t_tensor (tiempo)
            V_pred = model(Z_t, t_tensor)
            
            # 2. Paso de Euler: Z(t + dt) = Z(t) + v(Z(t), t) * dt
            Z_t = Z_t + V_pred * dt
            # GUARDAR: Almacena el estado de Z_t después de cada paso
            trajectories.append(Z_t.cpu().clone())
        # Z_t al final de la simulación es Z1
        Z1_generated = Z_t
        
    model.train() # Volver al modo entrenamiento
    return Z1_generated, torch.stack(trajectories, dim=0)

