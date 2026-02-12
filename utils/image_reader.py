import os
import numpy as np
from PIL import Image

'''

'''

def imageload(path, target_size=(64, 64)): # Reducido para eficiencia
    '''
    Docstring para imageload
    
    :param path: Ruta de la carpeta del cual se convertiran las imágenes a arrays de numpy
    :param target_size: redimensión de las imágenes
    '''
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
