import torch
import numpy as np
import matplotlib.pyplot as plt
from models.model import ConNet, Rectifiedflow
from train.train_rectified_flow import train_rectified_flow
from tqdm import tqdm

#Cargamos los datos:
path_noise = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\noise_subset_500.npy"
path_cartoon = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoon_subset_500.npy"

noise = np.load(path_noise)  # shape: (N, C, H, W)
cartoon = np.load(path_cartoon)  # shape: (N, C, H, W)
#Normalizamos las imágenes
noise = noise / 255.0 if noise.max() > 1 else noise
cartoon = cartoon / 255.0 if cartoon.max() > 1 else cartoon
#Convertimos a tensores de Pytorch
x0 = torch.tensor(noise, dtype=torch.float32).permute(0, 3, 1, 2)  # Cambiamos el orden a (N, C, H, W)
x1 = torch.tensor(cartoon, dtype=torch.float32).permute(0, 3, 1, 2)  # Cambiamos el orden a (N, C, H, W)


# 2. Construcción del modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConNet().to(device)
rectified_flow = Rectifiedflow(model=model, num_steps=100)

# 3. Entrenamiento
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batchsize = 64
inner_iters = 100

rectified_flow, loss_curve = train_rectified_flow(
    rectified_flow, optimizer, (x0, x1), batchsize, inner_iters, device=device
)

# 4. Visualización o guardado de resultados
plt.plot(loss_curve)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve Flow 1')
plt.savefig('loss_curve.png')
plt.show()

# Guardar el modelo
torch.save(model.state_dict(), "rectifiedflow_model.pth")
