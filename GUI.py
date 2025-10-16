import customtkinter as ctk
import tkinter as tk
import threading
import time

# --- Configuración Inicial ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")

class RectifiedFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuración de la Ventana Principal ---
        self.title("Rectified Flow: Herramienta de Flujo de Transporte")
        self.geometry("1100x750")
        self.grid_columnconfigure(0, weight=1) 
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Variables de estado (para simulación)
        self.is_training = False
        self.current_epoch = 0
        self.max_epochs = 100

        # Crear el Layout principal
        self.create_layout()

    def create_layout(self):
        # --- 1. Panel Lateral (Control y Parámetros) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        # Usamos CTkScrollableFrame para que el panel lateral pueda tener muchos parámetros
        # NOTA: Usaremos el parámetro 'label_text' solo aquí (el más robusto)
        self.control_panel = ctk.CTkScrollableFrame(self.sidebar_frame, label_text="⚙️ Configuración del Flujo")
        self.control_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.control_panel.grid_columnconfigure(0, weight=1)


        # --- 1.1. Selección del Dataset (pi_0 y pi_1) ---
        # SOLUCIÓN: Usamos un Frame simple con un Label como título.
        self.dataset_selector_label = ctk.CTkLabel(self.control_panel, text="1. Selección de Datasets", font=ctk.CTkFont(weight="bold"))
        self.dataset_selector_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.dataset_selector_frame = ctk.CTkFrame(self.control_panel) 
        self.dataset_selector_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")
        
        ctk.CTkLabel(self.dataset_selector_frame, text="Dataset $\\pi_0$ (Caras Reales):").pack(anchor="w", padx=10, pady=(5, 0))
        self.pi0_combo = ctk.CTkComboBox(self.dataset_selector_frame, values=["FFHQ", "CelebA"])
        self.pi0_combo.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(self.dataset_selector_frame, text="Dataset $\\pi_1$ (Caricaturas):").pack(anchor="w", padx=10, pady=(2, 0))
        self.pi1_combo = ctk.CTkComboBox(self.dataset_selector_frame, values=["Anime Faces", "Cartoon Set"])
        self.pi1_combo.pack(fill="x", padx=10, pady=(0, 15))


        # --- 1.2. Selección del Modelo y Parámetros ---
        self.param_label = ctk.CTkLabel(self.control_panel, text="2. Modelo y Parámetros", font=ctk.CTkFont(weight="bold"))
        self.param_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        self.param_frame = ctk.CTkFrame(self.control_panel)
        self.param_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.param_frame, text="Modelo (Arquitectura):").pack(anchor="w", padx=10, pady=(10, 0))
        self.model_combo = ctk.CTkComboBox(self.param_frame, values=["Modelo UNet V1", "Modelo Rectified V2"])
        self.model_combo.pack(fill="x", padx=10, pady=(0, 10))

        # Porcentajes (Train/Validation/Test)
        self.label_split = ctk.CTkLabel(self.param_frame, text="Split Datos (Train/Val/Test): 80 / 10 / 10")
        self.label_split.pack(anchor="w", padx=10, pady=(5, 0))
        
        self.slider_split = ctk.CTkSlider(self.param_frame, from_=50, to=90, number_of_steps=40)
        self.slider_split.set(80)
        self.slider_split.pack(fill="x", padx=10, pady=(0, 10))

        # Batch Size
        ctk.CTkLabel(self.param_frame, text="Batch Size:").pack(anchor="w", padx=10, pady=(2, 0))
        self.batch_entry = ctk.CTkEntry(self.param_frame, placeholder_text="Ej: 32")
        self.batch_entry.pack(fill="x", padx=10, pady=(0, 10))

        # Inner Iters
        ctk.CTkLabel(self.param_frame, text="Inner Iters (Subpasos de Flujo):").pack(anchor="w", padx=10, pady=(2, 0))
        self.iters_entry = ctk.CTkEntry(self.param_frame, placeholder_text="Ej: 100")
        self.iters_entry.pack(fill="x", padx=10, pady=(0, 15))


        # --- 2. Panel Principal (Visualización) ---
        self.main_panel = ctk.CTkTabview(self, segmented_button_fg_color="#333333", segmented_button_selected_color="#006400")
        self.main_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.main_panel.add("📊 Datos y PCA")
        self.main_panel.add("📈 Entrenamiento")
        self.main_panel.add("🖼️ Prueba y Resultados")

        # --- Pestaña 1: Datos y PCA ---
        self.data_tab = self.main_panel.tab("📊 Datos y PCA")
        self.data_tab.grid_columnconfigure((0, 1), weight=1)
        self.data_tab.grid_rowconfigure(1, weight=1)
        
        # 1.1 Visualización de Muestras
        # SOLUCIÓN: Frame simple + Label de título
        self.samples_title = ctk.CTkLabel(self.data_tab, text="Muestras Aleatorias (pi_0 y pi_1)", font=ctk.CTkFont(weight="bold"))
        self.samples_title.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.samples_frame = ctk.CTkFrame(self.data_tab)
        self.samples_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.samples_frame, text="[Placeholder: Mosaico de Imágenes de Ambos Datasets]").pack(fill="both", expand=True, padx=20, pady=20)

        # 1.2 Visualización de PCA
        # SOLUCIÓN: Frame simple + Label de título
        self.pca_title = ctk.CTkLabel(self.data_tab, text="Distribución PCA (2D)", font=ctk.CTkFont(weight="bold"))
        self.pca_title.grid(row=2, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.pca_frame = ctk.CTkFrame(self.data_tab)
        self.pca_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
        
        ctk.CTkLabel(self.pca_frame, text="[Placeholder: Gráfico Matplotlib/Canvas con los clusters $\\pi_0$ y $\\pi_1$ en 2D]").pack(fill="both", expand=True, padx=20, pady=20)


        # --- Pestaña 2: Entrenamiento ---
        self.training_tab = self.main_panel.tab("📈 Entrenamiento")
        self.training_tab.grid_columnconfigure(0, weight=1)
        self.training_tab.grid_rowconfigure(1, weight=1)
        
        # 2.1 Botón de Control y Barra de Progreso
        self.training_control_frame = ctk.CTkFrame(self.training_tab)
        self.training_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.training_control_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(self.training_control_frame, text="Estado: Listo.", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.grid(row=0, column=0, padx=10, pady=(5, 5), sticky="w")
        
        self.progress_bar = ctk.CTkProgressBar(self.training_control_frame, orientation="horizontal")
        self.progress_bar.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)

        self.start_button_main = ctk.CTkButton(self.training_control_frame, text="▶️ INICIAR ENTRENAMIENTO", command=self.start_training_thread, fg_color="green", hover_color="#006400")
        self.start_button_main.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="e")
        

        # 2.2 Gráficos de Loss
        self.loss_title = ctk.CTkLabel(self.training_tab, text="Loss (Pérdida) del Modelo", font=ctk.CTkFont(weight="bold"))
        self.loss_title.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        self.loss_chart_frame = ctk.CTkFrame(self.training_tab)
        self.loss_chart_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        ctk.CTkLabel(self.loss_chart_frame, text="[Placeholder: Gráfico Matplotlib de Loss vs. Época]").pack(fill="both", expand=True, padx=20, pady=20)


        # --- Pestaña 3: Prueba y Resultados ---
        self.results_tab = self.main_panel.tab("🖼️ Prueba y Resultados")
        self.results_tab.grid_columnconfigure((0, 1, 2), weight=1)
        self.results_tab.grid_rowconfigure(0, weight=1)

        # 3.1 Visualización de la Tripleta de Imágenes
        # Pi_0 (Real)
        self.pi0_title = ctk.CTkLabel(self.results_tab, text="Entrada ($\pi_0$)", font=ctk.CTkFont(weight="bold"))
        self.pi0_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.pi0_frame = ctk.CTkFrame(self.results_tab)
        self.pi0_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(self.pi0_frame, text="[Imagen $\\pi_0$ (Real)]").pack(fill="both", expand=True, padx=10, pady=10)
        
        # Pi_1 (Objetivo)
        self.pi1_title = ctk.CTkLabel(self.results_tab, text="Objetivo ($\pi_1$)", font=ctk.CTkFont(weight="bold"))
        self.pi1_title.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")
        self.pi1_frame = ctk.CTkFrame(self.results_tab)
        self.pi1_frame.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(self.pi1_frame, text="[Imagen $\\pi_1$ (Caricatura Real)]").pack(fill="both", expand=True, padx=10, pady=10)

        # Pi_1 Predicha
        self.pi1_pred_title = ctk.CTkLabel(self.results_tab, text="Predicha ($\pi_1$ Pred)", font=ctk.CTkFont(weight="bold"))
        self.pi1_pred_title.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="w")
        self.pi1_pred_frame = ctk.CTkFrame(self.results_tab)
        self.pi1_pred_frame.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(self.pi1_pred_frame, text="[Imagen Predicha (Resultado del Modelo)]").pack(fill="both", expand=True, padx=10, pady=10)

        # 3.2 Botones de Control
        self.control_buttons_frame = ctk.CTkFrame(self.results_tab)
        self.control_buttons_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.control_buttons_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.test_button = ctk.CTkButton(self.control_buttons_frame, text="🔄 Probar con Nueva Imagen")
        self.test_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.save_button = ctk.CTkButton(self.control_buttons_frame, text="💾 GUARDAR MODELO (Exportar)", fg_color="blue", hover_color="#00008b")
        self.save_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")


    # --- Funciones de Lógica de la GUI (Manejo de Hilos) ---

    def run_training(self):
        """
        [SIMULACIÓN DEL CÓDIGO PYTORCH PARA RECTIFIED FLOW]
        """
        self.is_training = True
        self.status_label.configure(text="Estado: 🧠 Entrenando... (Rectified Flow Activo)")
        self.progress_bar.set(0)
        
        # Tomar valor de las épocas del slider del panel lateral
        # Usamos 100 como valor máximo por defecto ya que el slider no tiene un callback de actualización aquí
        self.max_epochs = 100 
        
        # Simulación del bucle de entrenamiento
        for epoch in range(1, self.max_epochs + 1):
            if not self.is_training:
                break
            
            time.sleep(0.05) # Simulación de cómputo (Tu código PyTorch real iría aquí)

            # --- Actualización de la GUI ---
            progress_value = epoch / self.max_epochs
            self.progress_bar.set(progress_value)
            self.status_label.configure(text=f"Estado: Entrenando (Época {epoch}/{self.max_epochs}) - Loss: {0.5 / epoch:.4f}")
            self.current_epoch = epoch 
            

        # --- Finalización ---
        if self.is_training:
            self.status_label.configure(text="Estado: ✅ Entrenamiento Finalizado. Modelo Listo.")
            self.progress_bar.set(1.0)
        else:
            self.status_label.configure(text="Estado: 🛑 Entrenamiento Cancelado.")
        
        self.is_training = False
        self.start_button_main.configure(text="▶️ INICIAR ENTRENAMIENTO", command=self.start_training_thread, fg_color="green", hover_color="#006400", state="normal")

    def start_training_thread(self):
        """Inicia el proceso de entrenamiento en un hilo separado o lo cancela."""
        if self.is_training:
            # Cancelar
            self.is_training = False
            self.start_button_main.configure(text="Deteniendo...", state="disabled", fg_color="red")
        else:
            # Iniciar
            self.start_button_main.configure(text="⏸ CANCELAR ENTRENAMIENTO", command=self.cancel_training, fg_color="red", hover_color="#8b0000")
            
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()

    def cancel_training(self):
        """Lógica para la cancelación."""
        if self.is_training:
            self.is_training = False
            self.start_button_main.configure(text="Deteniendo...", state="disabled")

    def update_epoch_label(self, value):
        """Función placeholder para actualizar el valor del slider si fuera necesario."""
        self.max_epochs = int(value)
        self.label_epoch_value.configure(text=f"Valor: {self.max_epochs}")
        
if __name__ == "__main__":
    app = RectifiedFlowApp()
    app.mainloop()