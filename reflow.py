import torch
import numpy as np
import matplotlib.pyplot as plt
from models.model import ConNet, Rectifiedflow
from train.train_rectified_flow import train_rectified_flow

# Carga los datos originales
path_pi_0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\noise_subset_500.npy"
path_pi_1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoon_subset_500.npy"
pi_0 = np.load(path_pi_0)
pi_1 = np.load(path_pi_1)

x0 = torch.tensor(pi_0, dtype=torch.float32).permute(0, 3, 1, 2)
x1 = torch.tensor(pi_1, dtype=torch.float32).permute(0, 3, 1, 2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carga el primer modelo entrenado
model1 = ConNet().to(device)
model1.load_state_dict(torch.load("rectifiedflow_model.pth", map_location=device))
model1.eval()
rectified_flow1 = Rectifiedflow(model=model1, num_steps=100)

# Genera imágenes intermedias con el primer modelo
with torch.no_grad():
    x1_hat_list = []
    for i in range(x0.shape[0]):
        z0 = x0[i:i+1].to(device)
        traj = rectified_flow1.sample_ode(z0)
        x1_hat_list.append(traj[-1].cpu())
    x1_hat = torch.cat(x1_hat_list, dim=0)  # (N, C, H, W)

# Segundo modelo
model2 = ConNet().to(device)
rectified_flow2 = Rectifiedflow(model=model2, num_steps=100)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
batchsize = 64
inner_iters = 100

# Entrena el segundo rectified flow: de x1_hat a x1 (refinamiento)
rectified_flow2, loss_curve2 = train_rectified_flow(
    rectified_flow2, optimizer2, (x1_hat, x1), batchsize, inner_iters, device=device
)

plt.plot(loss_curve2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve Flow 2')
#plt.savefig('loss_curve flow 2.png')
plt.show()
# Guarda el segundo modelo
torch.save(model2.state_dict(), "rectifiedflow2_model.pth")