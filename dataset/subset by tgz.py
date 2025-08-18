import io
from io import BytesIO
import tarfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#Ruta del datset
path_tgz = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\cartoonset10k.tgz"
#Número de imágenes a extraer
m = 500 
#Dimensiones de las imágenes:
img_size = (64, 64)
#Convertimos a numpy y redimencionamos las imágenes
with tarfile.open(path_tgz, "r:gz") as tar:
    images_files = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

    #Tomamos archivos de forma aleatoria
    select_files = np.random.choice(images_files, size=m, replace=False)
    #Lista para almecenar las imágenes en numpy
    images=[]
    for member in select_files:
        f = tar.extractfile(member)
        if f is not None:
            img = Image.open(BytesIO(f.read())).convert("RGB").resize(img_size)
            images.append(np.array(img))

# Convertimos a array de NumPy final
images_np = np.array(images)
#Mostramos las diemnsiones del array
print(f"Dimensiones del array de imágenes: {images_np.shape}")
#salvamos:
np.save("cartoon_subset_500.npy", images_np)
