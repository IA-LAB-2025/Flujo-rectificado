import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters, device='cpu'):
    """
    Entrenamiento de la red de Rectified Flow para imágenes.

    Args:
        rectified_flow: Red neuronal (torch.nn.Module) que representa el flujo rectificado.
        optimizer: Optimizador de PyTorch.
        pairs: Tupla (x0, x1) donde:
            - x0: ruido gaussiano (tensor NxCxHxW)
            - x1: imágenes reales (tensor NxCxHxW)
        batchsize: Tamaño de batch para entrenamiento.
        inner_iters: Número de épocas de entrenamiento.
        device: 'cuda' o 'cpu'.
    """
    loss_curve=[]
    x0, x1 = pairs  # Ambos son tensores de imágenes (N, C, H, W)
    x0, x1 = x0.to(device), x1.to(device)
    dataset = TensorDataset(x0, x1)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    rectified_flow.model.train()
    for epoch in tqdm(range(inner_iters), desc="Epochs"):
        total_loss = 0.0
        for batch_x0, batch_x1 in dataloader:
            batch_x0, batch_x1 = batch_x0.to(device), batch_x1.to(device)

            # Muestras aleatorias de t en [0, 1], con misma forma que el batch (N, 1, 1, 1)
            t = torch.rand(batch_x0.size(0), 1, 1, 1, device=device)  # (batch,1,1,1)


            # Interpolación lineal entre x0 (ruido) y x1 (imagen)
            xt = (1 - t) * batch_x0 + t * batch_x1

            # Vector de dirección
            ut = (batch_x1 - batch_x0)

            # Evaluar red: la red predice v_theta(xt, t)
            pred = rectified_flow.model(xt, t)

            # Loss: MSE entre predicción y vector ut
            loss = torch.mean((pred - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_curve.append(avg_loss)
        tqdm.write(f"Epoch {epoch+1}/{inner_iters} - Loss: {avg_loss:.6f}")
        #print(f"Epoch {epoch+1}/{inner_iters} - Loss: {avg_loss:.6f}")
    return rectified_flow, loss_curve