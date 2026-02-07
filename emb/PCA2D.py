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

# Verificación de carga
if pi1_np.size == 0 or pi0_np.size == 0:
    print("Error: No se cargaron imágenes. Revisa las rutas.")
else:
    flat_pi1 = pi1_np.reshape(pi1_np.shape[0], -1)
    flat_pi0 = pi0_np.reshape(pi0_np.shape[0], -1)
    X = np.vstack((flat_pi1, flat_pi0))

    # --- PCA ---

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    varianza_total = np.sum(pca.explained_variance_ratio_)
    print(f"Varianza total ({n_components} componentes): {varianza_total*100:.2f}%")

    # --- VISUALIZACIÓN ---
    X_pca_pi1 = X_pca[:len(flat_pi1), :]
    X_pca_pi0 = X_pca[len(flat_pi1):, :]

    plt.figure(figsize=(10, 5))

    # Subplot 1: Dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca_pi1[:, 0], X_pca_pi1[:, 1], alpha=0.4, c='blue', label='pi_1', s=10)
    plt.scatter(X_pca_pi0[:, 0], X_pca_pi0[:, 1], alpha=0.4, c='red', label='pi_0', s=10)
    plt.title("Espacio Latente PCA (2D)")
    plt.legend()

    # Subplot 2: Varianza Acumulada
    plt.subplot(1, 2, 2)
    # Calculamos la varianza acumulada de los componentes que ya tenemos
    # (O podrías usar pca_full aquí si quieres ver hasta 100 componentes)
    pca_full = PCA(n_components=min(len(X), 100)).fit(X) 
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), color='green')
    plt.axhline(y=0.90, color='r', linestyle='--')
    plt.title("Varianza Acumulada")
    plt.xlabel("Componentes")

    plt.tight_layout()
    plt.show()