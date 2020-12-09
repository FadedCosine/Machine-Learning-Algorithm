import numpy as np
import pandas as pd

datalabel = np.array(["buying", "maint", "doors", "persons", "lug_boot", "safety", "values"])
feature2id = [{'vhigh': 0, 'high': 1, 'med': 2, 'low' : 3},
                    {'vhigh': 0, 'high': 1, 'med': 2, 'low' : 3},
                    {'2': 0, '3': 1, '4': 2, '5more': 3},
                    {'2': 0, '4': 1, 'more': 2},
                    {'small': 0, 'med': 1, 'big': 2},
                    {'low': 0, 'med': 1, 'high': 2},
                    {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}]         
id2feature = [ {i: w for w, i in feature_list.items()} for feature_list in feature2id]

def load_data(data_path, delimiter):
    data_sets = pd.read_csv(data_path, delimiter=delimiter, index_col=False, names=datalabel).to_numpy()
    np.random.shuffle(data_sets)
    return data_sets