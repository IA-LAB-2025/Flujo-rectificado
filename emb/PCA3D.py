from mpl_toolkits.mplot3d import Axes3D # Importante para habilitar 3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from PIL import Image

def imageload(path, target_size=(64, 64)): # Reducido para eficiencia
    imagenes = []
    if not os.path.exists(path):
        print(f"Error: La ruta {path} no existe.")
        return np.array([])
        
    for filename in os.listdir(path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                img = Image.open(os.path.join(path, filename)).convert('RGB')
                img = img.resize(target_size)
                imagenes.append(np.asarray(img) / 255.0)
            except Exception:
                continue # Ignora imágenes corruptas silenciosamente
    return np.array(imagenes)

# --- CARGA Y PROCESAMIENTO ---
path_pi1 = r"dataset/Caballo_Cebra/trainB"
path_pi0 = r"dataset/Caballo_Cebra/trainA"

pi1_np = imageload(path_pi1)
pi0_np = imageload(path_pi0)
n_components=3

if pi1_np.size == 0 or pi0_np.size == 0:
    print("Error: No se cargaron imágenes.")
else:
    flat_pi1 = pi1_np.reshape(pi1_np.shape[0], -1)
    flat_pi0 = pi0_np.reshape(pi0_np.shape[0], -1)
    X = np.vstack((flat_pi1, flat_pi0))

    # --- PCA de 3 componentes ---
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)

    varianza_total = np.sum(pca.explained_variance_ratio_)
    print(f"Varianza total (3 componentes): {varianza_total*100:.2f}%")

    # --- VISUALIZACIÓN ---
    X_pca_pi1 = X_pca[:len(flat_pi1), :]
    X_pca_pi0 = X_pca[len(flat_pi1):, :]

    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Dispersión 3D
    ax = fig.add_subplot(1, 2, 1, projection='3d') # Aquí activamos el 3D
    
    # Graficamos usando las 3 columnas: [:, 0], [:, 1] y [:, 2]
    ax.scatter(X_pca_pi1[:, 0], X_pca_pi1[:, 1], X_pca_pi1[:, 2], 
               alpha=0.4, c='blue', label='Caballos (pi_1)', s=10)
    ax.scatter(X_pca_pi0[:, 0], X_pca_pi0[:, 1], X_pca_pi0[:, 2], 
               alpha=0.4, c='red', label='Cebras (pi_0)', s=10)
    
    ax.set_title("Espacio Latente PCA (3D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()

    # Subplot 2: Varianza Acumulada
    ax2 = fig.add_subplot(1, 2, 2)
    pca_full = PCA(n_components=min(len(X), 100)).fit(X) 
    ax2.plot(np.cumsum(pca_full.explained_variance_ratio_), color='green')
    ax2.axhline(y=0.90, color='r', linestyle='--')
    ax2.set_title("Varianza Acumulada")
    ax2.set_xlabel("Componentes")

    plt.tight_layout()
    plt.show()