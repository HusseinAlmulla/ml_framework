from itertools import product

import mahalanobis
import numpy as np
import pandas as pd
from enn.enn import ENN
from scipy.spatial.distance import euclidean

from sklearn.model_selection import train_test_split, GridSearchCV

from hyper_parameters import model_to_params


def prepare_data():
    # data = pd.read_csv("tcyb-yerima-2777960-mm/drebin-215-dataset-5560malware-9476-benign.csv")
    data = pd.read_csv("tcyb-yerima-2777960-mm/malgenome-215-dataset-1260malware-2539-benign.csv")

    data = data.sample(frac=1)
    print(len(data))

    data['class'].mask(data['class'] == 'S', 1, inplace=True)  # get rid of string labels
    data['class'].mask(data['class'] == 'B', 0, inplace=True)

    end = -1
    x = data.iloc[:, :end]  # split input from output
    y = data.iloc[:, end:]
    x = x.values.tolist()  # convert from pandas to numpy
    x = np.asarray(x)
    y = y.values.tolist()
    y = np.asarray(y)
    y = y.reshape(y.shape[0], )  # acceptable shape by sklearn
    xx_train, xx_test, yy_train, yy_test = train_test_split(x, y, test_size=0.2, random_state=40)
    return xx_train, xx_test, yy_train, yy_test


x_train, x_test, y_train, y_test = prepare_data()

modelclasses = list()
for model in model_to_params:
    keys = list(model_to_params[model].keys())
    values = list(model_to_params[model].values())
    params_kwargs = [dict(zip(keys, v)) for v in product(*values)]
    modelclasses.append([model, params_kwargs])

for Model, params_list in modelclasses:

    for params in params_list:
        # print(set(don_not_go_together[LogisticRegression]), set(params.keys))

        model = Model(**params)
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print(e)
            continue
        score = model.score(x_test, y_test)
        # insights.append((modelname, model, params, score))
        # insights.append((str(model).split('(')[0], params, score))
        print((str(model).split('(')[0], params, score))
