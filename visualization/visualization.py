import matplotlib.pyplot as plt
import numpy as np

pi_1_np = np.load(r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_1\pi_1_np_10_50_50_gen.npy")
pi_0_np = np.load(r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_0\pi_0_np_10_50_50_gen.npy")
idx = 1

#Verificamos el rango de valores
img_array = pi_1_np[idx]
print(img_array.shape)
print(img_array.dtype)
print(img_array.min(), img_array.max())

# Visualizamos de acuerdo a idx
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(pi_1_np[idx])
plt.axis('off')
plt.title(f'Image at Index {idx}')
plt.subplot(1, 2, 2)
plt.imshow(pi_0_np[idx])
plt.axis('off')
plt.suptitle(f'Image and Noise at Index {idx}')
plt.show()