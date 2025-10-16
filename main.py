import torch
import numpy as np
import matplotlib.pyplot as plt
from models.model import ConNet, Rectifiedflow
from train.train_rectified_flow import train_rectified_flow
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import time

# =========================
# 1. Carga de datos
# =========================
path_pi_0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_0\pi_0_np_1000_100_100_gen.npy"
path_pi_1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_1\pi_1_np_1000_100_100_gen.npy"

pi_0= np.load(path_pi_0)  # shape: (N, C, H, W)
pi_1 = np.load(path_pi_1)  # shape: (N, C, H, W)

# =========================
# 2. Separar train / val
# =========================
pi0_train, pi0_val, pi1_train, pi1_val = train_test_split(
    pi_0, pi_1, test_size=0.2, random_state=42
)

# Convertir a tensores y permutar a (N,C,H,W)
x0_train = torch.tensor(pi0_train, dtype=torch.float32).permute(0,3,1,2)
x1_train = torch.tensor(pi1_train, dtype=torch.float32).permute(0,3,1,2)
x0_val   = torch.tensor(pi0_val, dtype=torch.float32).permute(0,3,1,2)
x1_val   = torch.tensor(pi1_val, dtype=torch.float32).permute(0,3,1,2)
# Verificar dimensiones
print("x0_train:", x0_train.shape)
print("x1_train:", x1_train.shape)
print("x0_val:", x0_val.shape)
print("x1_val:", x1_val.shape)

# =========================
# 3. Construcción del modelo
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConNet().to(device)
rectified_flow = Rectifiedflow(model=model, num_steps=100)

# =========================
# 4. Entrenamiento
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batchsize = 30
inner_iters = 100

# Entrenar solo con datos de entrenamiento
rectified_flow, loss_curve_train = train_rectified_flow(
    rectified_flow, optimizer, (x0_train, x1_train), batchsize, inner_iters, device=device
)

# =========================
# 5. Guardado
# =========================
#Ruta para Guardar el Modelo
results_dir = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\results"
# Nombre dinámico
# Adquirir la hora actual
current_time = time.strftime("%Y%m%d-%H%M%S")
filename = f"flow_model_{current_time}.pth"
# Guardar modelo en esa carpeta
model_path = os.path.join(results_dir, filename)
# Guardar usando la ruta completa
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_curve': loss_curve_train,
}, model_path)
# =========================
# 6. Evaluación
# =========================
plt.plot(loss_curve_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve Flow 1')
#Guardamos la grafica de la curva de pérdida
plt.savefig(os.path.join(results_dir, f'loss_curve_{current_time}.png'))
plt.show()