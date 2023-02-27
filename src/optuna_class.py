import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, learning_rate_schedule, EarlyStopping, ReduceLROnPlateau

from optuna.integration import PyTorchLightningPruningCallback

from sklearn.metrics import mean_squared_error

from open_class import Open
import demo


results = []

class Optuna_opt:
    """
    Esta classe é usada para otimizar um modelo de rede neural usando Optuna. O objetivo é minimizar o erro quadrático médio (MSE) do modelo de rede neural no conjunto de testes. 

    Inicializamos a classe Optuna_opt com um dataframe contendo os dados de treinamento e teste. 
    A função objective() é usada para definir os hiperparâmetros do modelo e treiná-lo usando os dados fornecidos.
    O erro quadrático médio é calculado com base nos resultados do conjunto de testes e retornado como a métrica a ser otimizada.
    Os resultados são armazenados em uma lista para fins posteriores.
    """

    def __init__(self, df):
        self.df = df


    df = Open.df_train_test(Open.read_file(demo.path), demo.target)

    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = df[0], df[1], df[2], df[3]

    def objective(trial):
        # Defina o número de unidades ocultas na camada oculta.
        num_hidden_units = trial.suggest_int('num_hidden_units', 5, 10, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

        # Defina a arquitetura da rede neural.
        model = Sequential()
        model.add(Dense(num_hidden_units, activation=activation))
        model.add(Dense(1, activation='linear'))

        # Compile o modelo.
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        callback = EarlyStopping(monitor='loss', patience=3, mode="min")

        # Treine o modelo.
        model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[callback])

        # Calcule o erro no conjunto de teste e retorne-a como a métrica a ser otimizada.
        y_pred = model.predict(x_test)
        mse = np.round(mean_squared_error(y_test, y_pred), 7)

        results.append({'num_hidden_units': num_hidden_units, 'activation': activation, 'optimizer': optimizer, 'value': mse})
        
        return mse