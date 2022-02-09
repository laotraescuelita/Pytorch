#Importar las librerias a utilizar.
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#Paso cero: Extraer y preparar los datos.
#Craer datos de simulación.
from sklearn.datasets import make_regression
X, y, coefficients = make_regression(n_samples = 100,
    n_features = 3,
    n_informative = 3,
    n_targets = 1,
    noise = 0.0,
    coef = True,
    random_state = 1)
print('Matriz \n', X[:3])
print('Vector \n', y[:3])

# Pasar los datos de numpy a troch
X_ = torch.from_numpy(X.astype(np.float32))
y_ = torch.from_numpy(y.astype(np.float32))
y_ = y.view(y.shape[0], 1)

filas, columnas = X.shape


#Escojer el módelo
tamaño_entrada = columnas
vector_salida = 1
model0 = nn.Linear(tamaño_entrada, vector_salida)

# Seleccion de función de costo y descenso del gradiente
learning_rate = 0.01

funcioncosto = nn.MSELoss()
gradiente = torch.optim.SGD(modelo.parameters(), learning_rate=learning_rate)  

# Entrenar el módelo
iteraciones = 100
for iteracion in range(iteraciones):    
    #propagación hacái adelante
    vector_predecido = modelo(X)
    costo_ = funcioncosto(vector_predecido, y)
    
    # propagación de regreso
    costo.backward()
    gradiente.step()

    # el gradiente se debe de reiniciar
    gradiente.zero_grad()

    if (iteracion+1) % 10 == 0:
        print(f'iteración: {iteracion+1}, costo = {costo.item():.4f}')

# visualizar
prediccion = model0(X).detach().numpy()

plt.plot(X_, y_, 'ro')
plt.plot(X_, prediccion, 'b')
plt.show()
