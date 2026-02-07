import matplotlib.pyplot as plt
from visualization.ode_solver import sample_rectified_flow

def plot1(X0,model,device, path_save=None):
    '''Esta función intenta gráficar la imágen original desde Z0=X0 en t=0
       y va mostrando las inferencias del algoritmo rectified flow con n_steps = [1,2,5,10,1000]
       X0 : imagen a gráficar
       model : modelo del cual se hace la inferencia  
    '''
    model.eval()
    n_steps = [1,2,5,10,100]
    resultados = []
    # Guardamos el original (t=0) para la primera columna
    img_original = (X0[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
    resultados.append(("Original", img_original))
    # 3. Correr la inferencia para cada n_pasos
    for n in n_steps:
      print(f"Calculando para {n} pasos...")
      # Solo nos interesa el resultado final z1_final
      z1_final, _ = sample_rectified_flow(model, X0, num_steps=n, device=device)
    
      # Procesar para mostrar
      img_res = (z1_final[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
      resultados.append((f"steps={n}", img_res))

      # 4. Visualización
    fig, axes = plt.subplots(1, len(resultados), figsize=(20, 8))

    for i, (titulo, img) in enumerate(resultados):
       axes[i].imshow(img)
       axes[i].set_title(titulo)
       axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot2(X0,model,device, n_steps, save=None):
    '''
    Docstring para plot2
    
    :param X0: imagen a inicial
    :param model: modelo 
    :param device: CPU o CUDA
    :param n_steps: número de pasos de Euler
    :param save: True → salva la gráfica 
    '''
    X0 = X0.to(device)
    # Solo nos interesa el resultado final z1_final
    z1_final, _ = sample_rectified_flow(model, X0, n_steps, device=device)
    # Procesar para mostrar
    img_res = (z1_final[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
    fig, ax = plt.subplots(1,2,figsize=(8,6))
    ax[0].imshow(X0[0].cpu().permute(1,2,0)* 0.5 + 0.5)
    ax[0].set_title("Original")
    ax[1].imshow(img_res)
    ax[1].set_title(f"Final con n = {n_steps}")

def plot3(X0,model,device, n_steps,save=None):
   return
