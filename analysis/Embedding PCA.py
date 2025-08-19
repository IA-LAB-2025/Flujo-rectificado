import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Cargamos los datasets
path_pi1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoon_subset_500.npy"
path_pi0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\noise_subset_500.npy"
pi1_np = np.load(path_pi1) #--> (N,H,W,C)
pi0_np = np.load(path_pi0) #--> (N,H,W,C)
# Aplanamos las imágenes:
flat_pi1 = pi1_np.reshape(pi1_np.shape[0], -1)  # (N, H*W*C)
flat_pi0 = pi0_np.reshape(pi0_np.shape[0], -1)  # (N, H*W*C)
# Combinamos los datasets para aplicar PCA
X = np.vstack((flat_pi1, flat_pi0))  # (2N, H*W*C)
# Aplicamos PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)  # (2N, 2)
# Separamos los datos PCA en imágenes y ruido
X_pca_images = X_pca[:flat_pi1.shape[0], :]  
X_pca_noise = X_pca[flat_pi1.shape[0]:, :]
# Graficar los puntos con miniaturas
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X_pca_images[:, 0], X_pca_images[:, 1], alpha=0.3,c='blue', label='Imágenes')
ax.scatter(X_pca_noise[:, 0], X_pca_noise[:, 1], alpha=0.3, c='red', label='Ruido')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.title("Visualización de imágenes en 2D con PCA")
plt.show()
