import os
import json
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import threading  # Para manejar el entrenamiento en un hilo separado
import sys
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Resources.Train import NeuralNetwork  # Asegúrate de que la clase NeuralNetwork está en esta ruta

# Datos de entrada y salida (ampliados)
X = [
    [17, 11, 8, 0, 3],
    [4, 11, 9, 8, 1],
    [30, 6, 4, 8, 5],
    [22, 8, 3, 2, 9],
    [13, 3, 6, 4, 6],
    [19, 11, 4, 7, 3],
    [17, 12, 7, 1, 8],
    [18, 5, 9, 5, 9],
    [12, 2, 6, 0, 7],
    [5, 5, 4, 9, 5],
    [5, 8, 0, 3, 5],
    [16, 3, 7, 9, 1],
    [21, 12, 3, 8, 8],
    [14, 2, 4, 5, 9],
    [27, 5, 9, 0, 4],
    [27, 2, 6, 3, 3],
    [27, 11, 0, 2, 5],
    [4, 12, 8, 9, 1],
    [10, 1, 9, 4, 3],
    [22, 3, 3, 1, 4]
]

Y = [
    [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0],
    [0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5],
    [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0],
    [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5],
    [0, 0, 0, 0, 0.5, 0, 1, 0, 0, 0],
    [0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0, 0],
    [0, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0],
    [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
    [0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
    [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5],
    [0.5, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0],
    [0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5],
    [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5],
    [0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
    [0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5],
    [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0.5],
    [0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0]
]

def entrenar_modelo(alpha, epochs, hidden_size, update_callback):
    nn = NeuralNetwork(input_size=5, hidden_size=hidden_size, output_size=10, learning_rate=alpha, epochs=epochs, error_threshold=0.01)
    
    for epoch in range(1, epochs + 1):
        nn.train(X, Y)  # Entrenar el modelo
        if epoch % 100 == 0 or epoch == epochs:  # Mostrar el error cada 100 épocas o en la última época
            total_error = nn.get_error_history()[-1]  # Obtener el último error de la lista
            update_callback(epoch, total_error)  # Llamar al callback para actualizar la consola

    # Guardar los datos de entrenamiento en un archivo JSON
    error_data = [{"epoch": i + 1, "error": error} for i, error in enumerate(nn.get_error_history())]
    with open("training_data.json", "w") as file:
        json.dump(error_data, file)

    return nn

class ConsoleRedirect(io.StringIO):
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def write(self, s):
        self.textbox.insert(tk.END, s)
        self.textbox.see(tk.END)

    def flush(self):
        pass

def graficar():
    global root, console_textbox

    def mostrar_entrenamiento():
        limpiar_interfaz()

        # Parte superior para imagen y título
        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Texto sobre la imagen
        texto_label = ctk.CTkLabel(
            frame_superior,
            text="RNA BACK-PROPAGATION (Entrenamiento)",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        texto_label.place(relx=0.5, rely=0.5, anchor="center")

        # Crear un marco para los parámetros de entrenamiento
        marco_datos = ctk.CTkFrame(root, height=300)
        marco_datos.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Contenedor para parámetros de entrenamiento y consola, en la misma línea
        parametros_frame = ctk.CTkFrame(marco_datos, height=260, width=250)
        parametros_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Scroll para los inputs
        scrollable_frame = ctk.CTkScrollableFrame(parametros_frame, width=380, height=260)
        scrollable_frame.pack(pady=5)

        # Campos de entrada
        ctk.CTkLabel(scrollable_frame, text="Tasa de Aprendizaje (Alpha):").pack(pady=5)
        alpha_entry = ctk.CTkEntry(scrollable_frame, width=150)
        alpha_entry.pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Número de Épocas:").pack(pady=5)
        epochs_entry = ctk.CTkEntry(scrollable_frame, width=150)
        epochs_entry.pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Cantidad de Neuronas (Capa Oculta):").pack(pady=5)
        hidden_size_entry = ctk.CTkEntry(scrollable_frame, width=150)
        hidden_size_entry.pack(pady=5)

        # Botón para entrenar el modelo
        entrenar_button = ctk.CTkButton(scrollable_frame, text="Entrenar Modelo", command=lambda: entrenar_y_mostrar(alpha_entry.get(), epochs_entry.get(), hidden_size_entry.get()))
        entrenar_button.pack(pady=10)

        # Contenedor para salida de consola
        console_frame = ctk.CTkFrame(marco_datos, height=280, width=450)
        console_frame.pack(side=tk.RIGHT, padx=5, pady=10)

        console_textbox = ctk.CTkTextbox(console_frame, height=280, width=430, font=ctk.CTkFont(size=12, weight='bold'))
        sys.stdout = ConsoleRedirect(console_textbox)
        console_textbox.pack(pady=5)
    
    def mostrar_aplicacion():
        limpiar_interfaz()

        # Parte superior para imagen y título
        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Texto sobre la imagen
        texto_label = ctk.CTkLabel(
            frame_superior,
            text="RNA BACK-PROPAGATION (Aplicación)",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        texto_label.place(relx=0.5, rely=0.5, anchor="center")

        # Crear un marco para la aplicación
        marco_datos = ctk.CTkFrame(root, height=300)
        marco_datos.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Gráfica de error vs épocas
        try:
            with open("training_data.json", "r") as file:
                training_data = json.load(file)

            epochs = [data["epoch"] for data in training_data]
            errors = [data["error"] for data in training_data]

            fig, ax = plt.subplots()
            ax.plot(epochs, errors, marker='o', linestyle='-', color='blue')
            ax.set_title("Error vs Épocas")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Error Total")
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=marco_datos)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except FileNotFoundError:
            ctk.CTkLabel(marco_datos, text="No se encontraron datos de entrenamiento.", font=ctk.CTkFont(size=16)).pack(pady=20)

    def entrenar_y_mostrar(alpha, epochs, hidden_size):
        try:
            alpha = float(alpha)
            epochs = int(epochs)
            hidden_size = int(hidden_size)

            # Función para actualizar la consola
            def update_console(epoch, total_error):
                console_textbox.insert(tk.END, f"Época {epoch}: Error Total = {total_error:.6f}\n")
                console_textbox.see(tk.END)  # Desplazar hacia abajo

            # Entrenamiento en un hilo separado
            thread = threading.Thread(target=entrenar_modelo, args=(alpha, epochs, hidden_size, update_console))
            thread.start()  # Iniciar el hilo

        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores válidos.")

    def limpiar_interfaz():
        for widget in root.grid_slaves():
            if widget.grid_info()['row'] != 0:  # Deja intacta la fila con los botones de selección
                widget.destroy()

    # Crear la ventana principal
    ctk.set_appearance_mode("dark")  # Modo oscuro
    ctk.set_default_color_theme("blue")  # Tema azul por defecto

    root = ctk.CTk()
    root.title("Adeline | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("800x600")

    # Configurar la grilla de la ventana principal
    root.grid_rowconfigure(0, weight=0)  # Fila de los botones de selección
    root.grid_rowconfigure(1, weight=0)  # Fila para frame_superior
    root.grid_rowconfigure(2, weight=1)  # Fila para main_scrollable_frame
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Botones de selección entre "Aplicación" y "Entrenamiento", siempre visibles
    frame_seleccion = ctk.CTkFrame(root)
    frame_seleccion.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    boton_aplicacion = ctk.CTkButton(frame_seleccion, text="Aplicación", command=mostrar_aplicacion)
    boton_aplicacion.pack(side="left", padx=20)

    boton_entrenamiento = ctk.CTkButton(frame_seleccion, text="Entrenamiento", command=mostrar_entrenamiento)
    boton_entrenamiento.pack(side="left", padx=20)

    # Iniciar la aplicación
    root.mainloop()

# Ejecutar la función principal
graficar()
