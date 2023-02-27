import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Open:
    """
    Esta classe Open contém dois métodos: read_file e df_train_test.

    O método read_file recebe um arquivo como parâmetro e lê o arquivo de acordo com sua extensão, podendo ser .csv ou .xlsx.
    O método também remove todos os valores nulos da base de dados.

    O método df_train_test recebe uma base de dados, uma variável-alvo e alguns parâmetros opcionais (test_size e random_state).
    Primeiro, ele normaliza os dados usando o MinMaxScaler da sklearn para que os valores estejam entre 0 e 1.
    Em seguida, separa a base de dados em conjuntos de treinamento e teste usando o train_test_split da sklearn.
    Por fim, retorna os conjuntos de treinamento e teste para x (variáveis independentes) e y (variável dependente).
    """
    
    def __init__(self, file):
        self.file = file


    def read_file(file):
        name_file, extension = os.path.splitext(file)

        if extension == ".csv":
            data = pd.read_csv(file)
        elif extension == ".xlsx":
            data = pd.read_excel(file, engine="openpyxl")
        
        data = data.dropna()
        
        return data

    def df_train_test(data, target, test_size=0.3, random_state=42):
        sc = MinMaxScaler(feature_range=(0,1))

        x = sc.fit_transform(data.drop([target], axis=1))
        y = sc.fit_transform(data[target].values.reshape(-1,1))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        return x_train, x_test, y_train, y_test