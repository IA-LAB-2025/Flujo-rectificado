import torch
import numpy as np
import matplotlib.pyplot as plt
from models.model import ConNet, Rectifiedflow
from train.train_rectified_flow import train_rectified_flow
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# =========================
# 1. Carga de datos
# =========================
path_pi_0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_0\pi_0_np_10_50_50_gen.npy"
path_pi_1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_1\pi_1_np_10_50_50_gen.npy"

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
batchsize = 1
inner_iters = 10

# Entrenar solo con datos de entrenamiento
rectified_flow, loss_curve_train = train_rectified_flow(
    rectified_flow, optimizer, (x0_train, x1_train), batchsize, inner_iters, device=device
)

# =========================
# 5. Evaluación
# =========================
plt.plot(loss_curve_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve Flow 1')
#plt.savefig('loss_curve.png')
plt.show()
# =========================
# 6. Guardado
# =========================
#Ruta para Guardar el Modelo
results_dir = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\results"
# Nombre dinámico
filename = "flow_model.pth"
# Guardar modelo en esa carpeta
model_path = os.path.join(results_dir, filename)
# Guardar usando la ruta completa
torch.save(model.state_dict(), model_path)
# =============================================================================================================
# 7. Refinamiento del modelo(reflow)
# =============================================================================================================
# Cargar el modelo entrenado previamente
model1 = ConNet().to(device)
model1.load_state_dict(torch.load(model_path, map_location= device))
rectified_flow_1 = Rectifiedflow(model=model1, num_steps=100)
# Generar imágenes intermedias con el modelo entrenado
with torch.no_grad():
    x1_hat_list = []
    for i in range(x0_train.shape[0]):
        z0 = x0_train[i:i+1].to(device)
        traj = rectified_flow_1.sample_ode(z0)
        x1_hat_list.append(traj[-1].cpu())
    x1_hat = torch.cat(x1_hat_list, dim=0)  # (N, C, H, W)
# Sedundo Modelo
model2 = ConNet().to(device)
rectified_flow2 = Rectifiedflow(model=model2, num_steps=100)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
batchsize = 1
inner_iters = 100
# Entrena el segundo rectified flow: de x1_hat a x1 (refinamiento)
rectified_flow2, loss_curve2 = train_rectified_flow(
    rectified_flow2, optimizer2, (x1_hat, x1_train), batchsize, inner_iters, device=device
)
# Visualización
plt.plot(loss_curve2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve Flow 2')
#plt.savefig('loss_curve flow 2.png')
plt.show()
# Guarda el segundo modelo
torch.save(model2.state_dict(), "rectifiedflow2_model.pth")
