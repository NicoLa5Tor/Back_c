import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs, error_threshold):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.error_threshold = error_threshold

        # Inicializar pesos y sesgos
        random.seed(0)
        self.W1 = [[random.gauss(0, 0.01) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [[0] for _ in range(hidden_size)]
        self.W2 = [[random.gauss(0, 0.01) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [[0] for _ in range(output_size)]

        self.error_history = []  # Historial de errores durante el entrenamiento

    def sigmoid(self, x):
        return 1 / (1 + (2.718281828459045 ** -x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def dot_product(self, A, B):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def add_bias(self, matrix, bias):
        return [[matrix[i][j] + bias[i][0] for j in range(len(matrix[0]))] for i in range(len(matrix))]

    def train(self, X, Y):
        # Normalizar entradas
        X_min = [min(col) for col in zip(*X)]
        X_max = [max(col) for col in zip(*X)]
        X_norm = [[(x - X_min[i]) / (X_max[i] - X_min[i]) for i, x in enumerate(row)] for row in X]

        # Transponer para muestras por columnas
        X_train = [list(x) for x in zip(*X_norm)]  # Dimensiones: (tamaño_entrada, número_muestras)
        Y_train = [list(y) for y in zip(*Y)]       # Dimensiones: (tamaño_salida, número_muestras)

        m = len(X_train[0])  # Número de muestras

        for epoch in range(1, self.epochs + 1):
            # Propagación hacia adelante
            Z1 = self.add_bias(self.dot_product(self.W1, X_train), self.b1)
            A1 = [[self.sigmoid(z) for z in row] for row in Z1]
            Z2 = self.add_bias(self.dot_product(self.W2, A1), self.b2)
            A2 = [[self.sigmoid(z) for z in row] for row in Z2]
            
            # Cálculo del error
            loss = (1 / (2 * m)) * sum((A2[i][j] - Y_train[i][j]) ** 2 for i in range(len(A2)) for j in range(len(A2[0])))
            self.error_history.append(loss)
            
            if epoch % 100 == 0 or loss <= self.error_threshold:  # Mostrar el error cada 100 épocas o si está por debajo del umbral
                print(f"Época {epoch}, Error: {loss}")

            # Condición de parada por error aceptado
            if loss <= self.error_threshold:
                print("Entrenamiento finalizado por alcanzar el umbral de error.")
                break
            
            # Retropropagación
            dZ2 = [[A2[i][j] - Y_train[i][j] for j in range(len(A2[0]))] for i in range(len(A2))]
            dW2 = [[(1 / m) * sum(dZ2[i][j] * A1[k][j] for j in range(len(A1[0]))) for k in range(len(A1))] for i in range(len(dZ2))]
            db2 = [[(1 / m) * sum(dZ2[i][j] for j in range(len(dZ2[0])))] for i in range(len(dZ2))]
            
            dA1 = self.dot_product(list(map(list, zip(*self.W2))), dZ2)
            dZ1 = [[dA1[i][j] * self.sigmoid_derivative(Z1[i][j]) for j in range(len(Z1[0]))] for i in range(len(Z1))]
            dW1 = [[(1 / m) * sum(dZ1[i][j] * X_train[k][j] for j in range(len(X_train[0]))) for k in range(len(X_train))] for i in range(len(dZ1))]
            db1 = [[(1 / m) * sum(dZ1[i][j] for j in range(len(dZ1[0])))] for i in range(len(dZ1))]
            
            # Actualizar pesos y sesgos
            self.W1 = [[self.W1[i][j] - self.learning_rate * dW1[i][j] for j in range(len(self.W1[0]))] for i in range(len(self.W1))]
            self.b1 = [[self.b1[i][0] - self.learning_rate * db1[i][0]] for i in range(len(self.b1))]
            self.W2 = [[self.W2[i][j] - self.learning_rate * dW2[i][j] for j in range(len(self.W2[0]))] for i in range(len(self.W2))]
            self.b2 = [[self.b2[i][0] - self.learning_rate * db2[i][0]] for i in range(len(self.b2))]

        # Mostrar pesos y sesgos finales
        print("\nPesos y Sesgos Finales:")
        print("W1:\n", self.W1)
        print("b1:\n", self.b1)
        print("W2:\n", self.W2)
        print("b2:\n", self.b2)

        # Mostrar error final
        final_error = (1 / (2 * m)) * sum((A2[i][j] - Y_train[i][j]) ** 2 for i in range(len(A2)) for j in range(len(A2[0])))
        print(f"\nError Final: {final_error}")

    def predict(self, X_input, X_min, X_max):
        # Normalizar entradas
        X_input_norm = [[(x - X_min[i]) / (X_max[i] - X_min[i]) for i, x in enumerate(row)] for row in X_input]
        X_input_norm = [list(x) for x in zip(*X_input_norm)]

        # Propagación hacia adelante
        Z1_new = self.add_bias(self.dot_product(self.W1, X_input_norm), self.b1)
        A1_new = [[self.sigmoid(z) for z in row] for row in Z1_new]  # Aquí row es correcto, ya que se refiere a las filas de Z1_new
        Z2_new = self.add_bias(self.dot_product(self.W2, A1_new), self.b2)
        Y_pred_new = [[self.sigmoid(z) for z in row] for row in Z2_new]  # Aquí se corrige para usar la variable correcta

        return [list(y) for y in zip(*Y_pred_new)]  # Devolver con muestras en filas


    def get_error_history(self):
        return self.error_history
