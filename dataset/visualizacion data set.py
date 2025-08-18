import matplotlib.pyplot as plt
import numpy as np
images_np = np.load(r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoon_subset_500.npy")
noise_np = np.load(r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\noise_subset_500.npy")
idx = 50
#Verificamos el rango de valores
img_array = images_np[idx]
print(img_array.shape)
print(img_array.dtype)
print(img_array.min(), img_array.max())

# Visualizamos de acuerdo a idx
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(images_np[idx])
plt.axis('off')
plt.title(f'Image at Index {idx}')
plt.subplot(1, 2, 2)
plt.imshow(noise_np[idx])
plt.axis('off')
plt.suptitle(f'Image and Noise at Index {idx}')
plt.show()