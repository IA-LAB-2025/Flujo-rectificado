# El objetivo es cargar los datos y generar las distribuciones de imágenes
# Importamos las librerpias necesarias
import os
import random
from PIL import Image
import numpy as np
'''
Cartoonset10k: C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\dataset sin seleccionar\cartoonset10k\cartoonset10k
Celab: C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\dataset sin seleccionar\img_align_celeba\img_align_celeba

'''
# ========================== Variables ===========================
# Rutas de los datos
dataset_pi_0 = None # Aquí se define si se quiere generar imágenes o cambiar de dominio
dataset_pi_1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\dataset sin seleccionar\cartoonset10k\cartoonset10k"
# Redimensión de las imágenes
H, W = 50, 50 # Altura y Ancho de las imágenes
# Número de imágenes que se quieran en el dataset
#Por el mometo lo máximo es de 10,000 Imégenes
N = 10
#Ruta donde se guardarán los datos pi_0
path_pi_0 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_0"
#Ruta donde se guardarán los datos pi_1
path_pi_1 = r"C:\Users\gilda\Desktop\VS\Python\Flujo Rectificado Propuesta\dataset\pi_1"
# =========================== Funciones ==========================
# El objetivo es proporcionar las distribuciones pi_0 y pi_1 para clcular 
# el flujo rectificado
#Definimos la función load_data
#Primero definamos si se quiere Generar imágenes O Cambiar Dominio

def load_data(dataset_pi_0,dataset_pi_1,N, H, W,path_pi_0 ,path_pi_1 ):
    '''Cargamos las imágenes de las distribuciones pi_0 y pi_1
    - dataset_pi_0: ruta de las imágenes en pi_0 , si es None se generará imágenes de ruido
    - dataset_pi_1: ruta de las imágenes en pi_1
    - N: Número de imágenes a cargar, máximo 10,000
    - H: Altura a redimensionar
    - W: Ancho a redimensionar
    - path_pi_0: Ruta donde se guardarán los datos pi_0
    - path_pi_1: Ruta donde se guardarán los datos pi_1
    
    '''
    def load_random_images(path, N, H, W):
        '''Función para cargar N imágenes aleatorias de una carpeta
        redimensionamos a HxW y las normalizamos a [0,1]'''
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        # Obtemos las listas de los archivos
        all_images = [f for f in os.listdir(path) if f.lower().endswith(valid_exts)]
        # Seleccionamos N imágenes aleatorias
        selected_images = random.sample(all_images, N)
        image_array = []
        for image_name in selected_images:
            image_path = os.path.join(path, image_name)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((W, H))  # Redimensionamos la imagen WxH
            image_np = np.array(image)/255.0 #Normalizamos a [0,1]
            image_array.append(image_np)
        return np.stack(image_array)
    
    if dataset_pi_0 is None:
        print("Se ha seleccionado generación de imágenes")
        # =========================== Generamos las imágenes de Ruido(pi_0) =========================
        noise_np = np.random.normal(loc=0.5, scale=0.2, size=(N, H, W, 3)) # H, W, 3
        noise_np = np.clip(noise_np, 0, 1)  # Aseguramos que los valores estén en [0, 1]
        #Nombre dinámico del archivo
        filename = f"pi_0_np_{N}_{H}_{W}_gen.npy"
        save_path = os.path.join(path_pi_0, filename)
        #Guardamos el array  
        np.save(save_path,noise_np)
        print(f"Imágenes de ruido generadas y guardadas en {save_path}")
        print(f"Las Dimensiones del array de ruido {filename} son: {noise_np.shape}")
        #=========================== Generamos las imágenes de pi_1 ==================================
        # uso de la función load_random_images
        target_np = load_random_images(dataset_pi_1, N, H, W)
        #Nombre dinámico del archivo
        filename = f"pi_1_np_{N}_{H}_{W}_gen.npy"
        save_path = os.path.join(path_pi_1,filename)
        #Guardamos el array
        np.save(save_path, target_np)
        print(f"Imágenes de cartoon generadas y guardadas en {save_path}")
        print(f"Las Dimensiones del array de cartoon {filename} son: {target_np.shape}")
        
    else:
        #Entonces tendremos un cambio de dominio
        print("Se ha seleccionado un cambio de dominio")
        # =========================== Cargamos las imágenes de pi_0 =========================
        pi_0_np = load_random_images(dataset_pi_0,N, H, W)
        # Nombre dinámico del archivo
        filename = f"pi_0_np_{N}_{H}_{W}_dom.npy"
        save_path = os.path.join(path_pi_0, filename)
        # Guardamos el array 
        np.save(save_path, pi_0_np)
        print(f"Imágenes de pi_0 cargadas y guardadas en {save_path}")
        print(f"Las Dimensiones del array de pi_0 {filename} son: {pi_0_np.shape}")
        # =========================== Cargamos las imágenes de pi_1 =========================
        pi_1_np = load_random_images(dataset_pi_1, N, H, W)
        # Nombre dinámico del archivo
        filename = f"pi_1_np_{N}_{H}_{W}_dom.npy"
        save_path = os.path.join(path_pi_1, filename)
        # Guardamos el array 
        np.save(save_path, pi_1_np)
        print(f"Imágenes de pi_0 cargadas y guardadas en {save_path}")
        print(f"Las Dimensiones del array de pi_0 {filename} son: {pi_1_np.shape}")
        
##### Test #####
load_data(dataset_pi_0, dataset_pi_1, N, H, W, path_pi_0, path_pi_1)
# Fin del script