import optuna
import pandas as pd
import optuna_class as optuna_class

from optuna_class import Optuna_opt
from search_class import Search
from icecream import ic

import warnings
warnings.filterwarnings("ignore")


path = "data\Caso1_230kV_0.9_1.1.xlsx"
target = 'sigma'

if __name__ == '__main__':

    # Testando a otimização de hiperparâmetros pela biblioteca optuna
    """
    study = optuna.create_study(direction='minimize')
    ic(study.optimize(Optuna_opt.objective, n_trials=5))
    
    df_result = pd.DataFrame(optuna_class.results)
    ic(df_result.sort_values('value'))
    """


    # Testando a otimização por Grid Search
    """
    layer_config_list = [
        [(5, "relu"), (1, "linear")],
        [(10, "relu"), (1, "linear")]
    ]
    optimizer_list = ['Adam', 'SGD']
    epochs_list = [100, 500]

    ic(Search.search(layer_config_list=layer_config_list, optimizer_list=optimizer_list, epochs_list=epochs_list))
    """


    # Testando manualmente
    """
    layer_config = [[(5, "relu"), (1, "linear")]]
    optimizer = ["Adam"]
    epochs = [500]

    ic(Search.search(layer_config_list=layer_config, optimizer_list=optimizer, epochs_list=epochs))
    """