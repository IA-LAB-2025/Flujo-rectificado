import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargamos los datasets
path_pi1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_1\pi_1_np_10_50_50_gen.npy"
path_pi0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_0\pi_0_np_10_50_50_gen.npy"

pi1_np = np.load(path_pi1)  # (N, H, W, C)
pi0_np = np.load(path_pi0)  # (N, H, W, C)

# Aplanamos las imágenes
flat_pi1 = pi1_np.reshape(pi1_np.shape[0], -1)
flat_pi0 = pi0_np.reshape(pi0_np.shape[0], -1)

# Combinamos datasets
X = np.vstack((flat_pi1, flat_pi0))

# Paso 1: PCA a 50 dimensiones
pca_50 = PCA(n_components=10, random_state=42)
X_pca50 = pca_50.fit_transform(X)

# Paso 2: t-SNE a 2D
# Para versiones recientes de sklearn:
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init='pca',
    random_state=42
)

# Ejecutar t-SNE (con max_iter si tu versión lo soporta)
X_tsne = tsne.fit_transform(X_pca50)

# Separar caricaturas y ruido
X_tsne_images = X_tsne[:flat_pi1.shape[0], :]
X_tsne_noise = X_tsne[flat_pi1.shape[0]:, :]

# Graficar
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X_tsne_images[:, 0], X_tsne_images[:, 1], alpha=0.3, c='blue', label='Imágenes')
ax.scatter(X_tsne_noise[:, 0], X_tsne_noise[:, 1], alpha=0.3, c='red', label='Ruido')
plt.xlabel("t-SNE dimensión 1")
plt.ylabel("t-SNE dimensión 2")
plt.legend()
plt.title("Visualización de imágenes en 2D con t-SNE")
plt.show()
