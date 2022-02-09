import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

filas, columnas = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)


iteraciones = 100
learning_rate = 0.01
funcioncosto = nn.BCELoss()
gradiente = torch.optim.SGD(model.parameters(), learning_rate=learning_rate)


for iteracion in range(iteraciones):
    
    y_pred = model(X_train)
    costo = funcioncosto(y_pred, y_train)

    
    costo.backward()
    gradiente.step()

    
    gradiente.zero_grad()

    if (iteracion+1) % 10 == 0:
        print(f'epoch: {iteracion+1}, loss = {costo.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')