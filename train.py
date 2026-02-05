import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import argparse
import os
import json
from tqdm import tqdm

# Importaciones de tus módulos
from models.model import UNET
from dataset.dataset import ImageDataset

# =========================
#  Carga de datos
# =========================
'''
PATH_PI_0 : Es la ruta donde se encuentran las imágenes de inicio (X0) ~ π_0
PATH_PI_1 : Es la ruta donde se encuentran las imágenes objetivo  (X1) ~ π_1
'''
PATH_PI_0 = r"dataset/PetImages/Cat"
PATH_PI_1 = r"dataset/cats_projekat/salvajes"

# =========================
#  Parámetros
# =========================
def get_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de Rectified Flow")
    # Parámetros del Dataset
    parser.add_argument('--path_a', type=str, default=PATH_PI_0)
    parser.add_argument('--path_b', type=str, default=PATH_PI_1)
    parser.add_argument('--limit', type=int, default=None, help="Límite de imágenes para pruebas")
    
    # Hiperparámetros
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--channels', type=int, default=64, help="Canales base de la UNET")
    parser.add_argument('--exp_name', type=str,default='model_rectified_flow')
    return parser.parse_args()

# =========================
#  Funcines de Entrenamiento
#  y Validación
# =========================
def train_one_epoch(model,dataloader, optimizer, device):
    """
    Ejecuta una época de entrenamiento para el modelo Rectified Flow.
    
    Args:
        model (nn.Module): La U-Net acondicionada por tiempo.
        dataloader (DataLoader): DataLoader con pares (X0, X1).
        optimizer (optim.Optimizer): Optimizador (ej. Adam).
        device (torch.device): Dispositivo (CPU o CUDA).
        
    Returns:
        float: La pérdida promedio de la época.
    """
    model.train() # Pone el modelo en modo entrenamiento
    total_loss = 0.0
    for batch in dataloader:
        #mandamos a cpu o gpu
        X0_batch = batch["X0"].to(device)
        X1_batch = batch["X1"].to(device)
        # --- 2. MUESTREO DE TIEMPO (t) ---
        # t debe ser un tensor 1D (B) con valores en [0, 1]
        t = torch.rand(X0_batch.size(0), device=device)
        # --- 3. CÁLCULO DE LA TRAYECTORIA Y EL OBJETIVO ---
        
        # Trayectoria Ideal (Target): X1 - X0
        V_target = X1_batch - X0_batch
        #Calculamos el punto intermedio X_t
        # Punto Intermedio (X_t): Xt = t*X1 + (1-t)*X0
        # Expandimos t de (B) a (B, 1, 1, 1) para el broadcasting con la imagen (B, C, H, W)
        t_view = t.view(-1, 1, 1, 1) 
        X_t = t_view * X1_batch + (1 - t_view) * X0_batch
        # --- 4. MODELO Y PÉRDIDA ---
        optimizer.zero_grad()
        
        # Obtener la velocidad predicha v(X_t, t)
        V_pred = model(X_t, t)
        # Pérdida de Rectified Flow: MSE(V_pred, V_target)
        # Esto implementa la minimización de E[||(X1 - X0) - v(X_t, t)||^2]
        loss = F.mse_loss(V_pred, V_target)
        # --- 5. OPTIMIZACIÓN ---
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X0_batch.size(0)
    # Devolvemos la pérdida promedio para toda la época
    return total_loss / len(dataloader.dataset)
def validate_one_epoch(model, dataloader, device):
    """
    Calcula la pérdida del Rectified Flow sobre el conjunto de validación.
    
    Args:
        model (nn.Module): La U-Net acondicionada por tiempo.
        dataloader (DataLoader): DataLoader con pares (X0, X1) de validación.
        device (torch.device): Dispositivo (CPU o CUDA).
        
    Returns:
        float: La pérdida promedio de validación de la época.
    """
    model.eval() # Ponemos en modo evaluación para no afectar capas como BatchNorm o Dropout
    total_loss = 0
    
    # CRÍTICO: Desactiva el cálculo de gradientes (ahorra memoria y tiempo)
    with torch.no_grad():
        for batch in dataloader:
            
            X0_batch = batch["X0"].to(device)
            X1_batch = batch["X1"].to(device)
            
            # --- 1. MUESTREO DE TIEMPO (t) ---
            # Aunque la validación podría usar pasos fijos, es estándar samplear t ~ U[0, 1] 
            # igual que el entrenamiento para una comparación directa del objetivo de pérdida.
            t = torch.rand(X0_batch.size(0), device=device)
            
            # --- 2. CÁLCULO DE LA TRAYECTORIA Y EL OBJETIVO ---
            V_target = X1_batch - X0_batch 
            
            t_view = t.view(-1, 1, 1, 1) 
            X_t = t_view * X1_batch + (1 - t_view) * X0_batch
            
            # --- 3. MODELO Y PÉRDIDA ---
            V_pred = model(X_t, t) 

            # Pérdida de Rectified Flow (MSE)
            loss = F.mse_loss(V_pred, V_target)
            
            total_loss += loss.item() * X0_batch.size(0)

    # Volvemos a poner el modelo en modo entrenamiento después de la validación
    model.train()
    
    return total_loss / len(dataloader.dataset)
# =========================
#  Funcion Main
# =========================

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Trabajando en: {device}")

    # 1. Preparar Carpeta de Resultados
    save_path = os.path.join("results", args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    # 2. Transformaciones
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 3. Dataset y Dataloader
    full_ds = ImageDataset(args.path_a, args.path_b, transform, max_samples=args.limit)
    
    # División Train/Val (90% - 10%)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Modelo y Optimización
    model = UNET(channels_in=3, channels=args.channels, channels_out=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_history = {'train': [], 'val': []}

    # 5. Loop Principal
    pbar = tqdm(range(args.epochs), desc="Progeso Total")
    for epoch in pbar:
        #Entrenamiento
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        loss_history['train'].append(train_loss)
        #Validation
        val_loss = validate_one_epoch(model, val_loader, device)
        loss_history['val'].append(val_loss)
        # Actualizar la barra de progreso con la pérdida actual
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})      
        print(f"Época [{epoch+1}/{args.epochs}] - Loss: {train_loss:.6f}")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época [{epoch+1}/{args.epochs}] | Pérdida Entrenamiento: {train_loss:.6f} | Pérdida Validación: {val_loss:.6f}")
    
    checkpoint_path = os.path.join(save_path, f"{args.exp_name}_final.pth")
    torch.save(model.state_dict(), checkpoint_path)
    # Guardar el historial de pérdidas en un JSON para graficar después
    with open(os.path.join(save_path, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)
    print(f"Modelo guardado exitosamente en: {checkpoint_path}")

if __name__ == "__main__":
    main()