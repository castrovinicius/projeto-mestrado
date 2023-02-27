import numpy as np
import time

from itertools import product
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, learning_rate_schedule, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import mean_squared_error

from open_class import Open
import demo

class Search:
    """
    Esta classe implementa um algoritmo de busca para encontrar a melhor configuração de camadas, otimizador e épocas para um modelo de rede neural.
    
    O construtor da classe recebe como parâmetros os dados, a lista de configurações de camadas, a lista de otimizadores e a lista de épocas.
    O método search() usa esses parâmetros para criar um modelo sequencial e compilá-lo com os parâmetros especificados.
    Um callback EarlyStopping é usado para interromper o treinamento se o erro não diminuir por três iterações consecutivas.
    Os dados são divididos em conjuntos de treinamento e teste e o modelo é treinado com os dados de treinamento.
    O tempo total necessário para treinar o modelo é exibido na saída do programa, bem como a pontuação da precisão do modelo (calculada usando erro médio quadrático).
    """

    def __init__(self, data, layer_config_list, optimizer_list, epochs_list):
        self.data = data
        self.layer_config_list = layer_config_list
        self.optimizer_list = optimizer_list
        self.epochs_list = epochs_list


    def search(layer_config_list, optimizer_list, epochs_list):
        data = Open.df_train_test(Open.read_file(demo.path), demo.target)
        
        try:
            for layer_config, optimizer, epochs in product(layer_config_list, optimizer_list, epochs_list):
                print("-"*50)
                print(f"layer_config: {layer_config}")
                print(f"optimizer: {optimizer}")
                print(f"epochs: {epochs}")
                
                model = Sequential()
                for layer in layer_config:
                    model.add(Dense(layer[0], activation=layer[1]))
                
                model.compile(loss="mean_squared_error", optimizer=optimizer)

                callback = EarlyStopping(monitor='loss', patience=3, mode="min")
                
                x_train, x_test, y_train, y_test = data[0], data[1], data[2], data[3]
                start_time = time.time()
                history = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[callback])
                y_pred = model.predict(x_test)
                stop_time = time.time()

                print("-"*50)
                print("Training time:", np.round((stop_time - start_time), 2),"s")
                print("-"*50)
                print(f"Accuracy score: {np.round(mean_squared_error(y_test, y_pred), 5)}")
                print("\n"+"="*50+"\n")

        except Exception as e:
            print("exceção tratada: " + str(e))
            