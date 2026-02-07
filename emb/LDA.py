from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
# LDA sí necesita saber quién es quién (las etiquetas 'y')
y = np.array([1] * len(flat_pi1) + [0] * len(flat_pi0)) # 1 para Caballos, 0 para Cebras

# --- LDA ---
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# --- VISUALIZACIÓN MEJORADA ---
plt.figure(figsize=(10, 4))

# Generamos un "jitter" (valores aleatorios pequeños en Y) para separar los puntos visualmente
jitter_caballos = np.random.normal(0, 0.02, size=sum(y==1))
jitter_cebras = np.random.normal(0, 0.02, size=sum(y==0))

plt.scatter(X_lda[y==1], jitter_caballos, alpha=0.5, label='Caballos (pi_1)', c='blue', s=15)
plt.scatter(X_lda[y==0], jitter_cebras, alpha=0.5, label='Cebras (pi_0)', c='red', s=15)

# Estética de la gráfica
plt.axhline(0, color='black', linestyle='--', alpha=0.3) # Línea base
plt.title("Máxima Separación Lineal entre Caballos y Cebras (LDA)")
plt.xlabel("Eje Discriminante (Combinación óptima de píxeles)")
plt.yticks([]) # Quitamos los números del eje Y porque no significan nada
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()