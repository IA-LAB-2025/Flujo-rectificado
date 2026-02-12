from sklearn.manifold import TSNE
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

# 1. Reducción previa con PCA (Obligatorio para velocidad y ruido)
pca_pre = PCA(n_components=50, random_state=42)
X_pca_50 = pca_pre.fit_transform(X)

# 2. Aplicar t-SNE
# n_components=2 para verlo en un plano
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca_50)

# 3. Separar para graficar
X_tsne_pi1 = X_tsne[:len(flat_pi1), :] # Caballos
X_tsne_pi0 = X_tsne[len(flat_pi1):, :] # Cebras

# 4. Visualización
plt.figure(figsize=(10, 7))
plt.scatter(X_tsne_pi1[:, 0], X_tsne_pi1[:, 1], c='blue', label='Caballos', alpha=0.6, s=15)
plt.scatter(X_tsne_pi0[:, 0], X_tsne_pi0[:, 1], c='red', label='Cebras', alpha=0.6, s=15)

plt.title("Visualización No Lineal con t-SNE")
plt.xlabel("t-SNE eje 1")
plt.ylabel("t-SNE eje 2")
plt.legend()
plt.show()