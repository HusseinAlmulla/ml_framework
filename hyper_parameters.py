import manhattan as manhattan
from scipy.spatial.distance import minkowski
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from enn.enn import ENN

from distance_function import HasD, LD

model_to_params = {
    LogisticRegression: {
        "penalty": ['l1', 'l2', 'elasticnet'],
        "C": [0.01, 0.1, 0.2],
        "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    },
    SVC: {
        "C": [0.01, 0.1, 0.2],
    },
    GaussianNB: {},

    https://github.com/HusseinAlmulla/ENN
    ENN: {
        "k": [1, 3, 5]
    },
    KNeighborsClassifier: {
        "n_neighbors": [3],
        "metric": ['dice', HasD, 'rogerstanimoto']
    },

}
