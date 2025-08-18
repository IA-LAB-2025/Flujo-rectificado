import numpy as np
import matplotlib.pyplot as plt
# Generación de ruido gaussiano y visualización de imágenes
# Cargamos las imágenes para extraer el rango de valores
images_np = np.load(r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoon_subset_500.npy")  # forma: (500, 64, 64, 3)

# Normalizamos a [0, 1] si aún no están en ese rango
if images_np.max() > 1.0:
    images_np = images_np / 255.0
# Generamos ruido gaussiano con media 0.5 y desviación estándar 0.2
# Para que esté en [0, 1] la mayor parte del tiempo
noise_np = np.random.normal(loc=0.5, scale=0.2, size=images_np.shape)

# Clip a [0, 1]
noise_np = np.clip(noise_np, 0.0, 1.0)
# Guardamos en archivo .npy
np.save("noise_subset_500.npy", noise_np)