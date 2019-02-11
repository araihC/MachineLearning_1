

#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sn
from collections import OrderedDict

#Librerie per il sampling (imbalanced) dei dati
from imblearn.datasets import make_imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

from sklearn import datasets
#Librerie per l'allenamento DT
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd


#Librerie per il preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import QuantileTransformer

#Librerie per l'evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

#Librerie per il plot
from mlxtend.plotting import plot_decision_regions
from pandas_ml import ConfusionMatrix
#----------------------------Set params and options-----------------------------
#----------------------------- Parameter definition ----------------------------
path = '/Users/Mac/Desktop/ML/ML_project/'
path_data = path + 'Cleaned_data/'
file_name = path_data + 'CARRIERA.xlsx'
sheet_name = 'Minimal_4'
evaluation_file_name = 'Evaluation.txt'
path_results = path + 'Results/'+ sheet_name + '/RF/'

if not os.path.exists(path_results):
   os.makedirs(path_results)

CRITERIO = 'entropy' # Criterio del DT 'entropy' o 'gini'
N_SPLITS = 6

N_TREES = 100
MAX_DEPTH = 7
TEST_SIZE = 0.2
METHOD = 'KFold' #Random o KFold
SCALING_Method = 'Standard' #MinMax,Standard, Norm

title = 'RandomForest'

#===============================================================================

#===============================================================================

#-------------------------------------Loading dei dati--------------------------

#Loading dei dati
def DATA_CLEANING(dataset):
    X = dataset.drop('Anni_di_studio', axis = 1)
    X.drop('CFU', axis = 1, inplace = True)
    X.drop('Media_Pesata', axis = 1, inplace = True)
    X.drop('Media_Zona', axis = 1, inplace = True)
    Y = dataset['Media_Zona']

    #Prima di procedere devo fare l'encoding delle colonne contenenti stringhe
    #poiche' l'lgoritmo non accetta str come tipo
    values = np.array(X.REGIONE_ISTITUTO_DIPLOMA)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    X.drop('REGIONE_ISTITUTO_DIPLOMA', axis = 1, inplace = True)
    X['REGIONE_ISTITUTO_DIPLOMA'] = integer_encoded

    values = np.array(X.Tipo_Diploma)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    X.drop('Tipo_Diploma', axis = 1, inplace = True)
    X['Tipo_Diploma'] = integer_encoded

    values = np.array(X.Sesso)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    X.drop('Sesso', axis = 1, inplace = True)
    X['Sesso'] = integer_encoded

    #X.drop('Tipo_Diploma', axis = 1, inplace = True)
    #X.drop('Sesso', axis = 1, inplace = True)
    #CAMBIO
    X.drop('Data_nascita', axis = 1, inplace = True)
    #X.drop('REGIONE_ISTITUTO_DIPLOMA', axis = 1, inplace = True)
    #X.drop('Corso', axis = 1, inplace = True)

    values = np.array(Y)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #Y.drop('Media_Zona', axis = 1, inplace = True)
    Y = integer_encoded
    return Y, X

dataset = pd.read_excel(io=file_name, sheet_name=sheet_name)
Y, X = DATA_CLEANING(dataset)

#Valuatiamo ora se sono bilanciate o meno le classi


name = 'Pre_Bilanciamento.pdf'

X_pre = X
Y_pre = Y
sm = SMOTE(random_state=42, kind = 'svm', sampling_strategy = 'not majority')#'minority') #in questo modo sono 50%, 25%, 25%
X, Y = sm.fit_resample(X, Y)
name = 'Post_Bilanciamento.pdf'

#-----------------------Implementazione dell'algoritmo--------------------------

#Divido il dataset in test e train
if METHOD == 'Random':
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state = 100)

elif METHOD == 'KFold':
    kf = StratifiedKFold(n_splits = N_SPLITS, random_state = 100, shuffle = False)
    kf.get_n_splits(X, Y)
    for train_index, test_index in kf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]


def N_estimator_Evaluation(X_train, X_test, Y_train, Y_test, path_results, title):
    #alleno il classifier

    ensemble_clfs = [
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=100)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=100)),
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, oob_score=True,
                                   max_features=1,
                                   random_state=100))
    ]


    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 10
    max_estimators = 200

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, Y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        print(label)
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    '''
    xs = []
    ys = []
    for label, clf_err in error_rate.items():
        x, y = zip(*clf_err)
        xs.append(x)
        ys.append(y)

    plt.plot(xs[0], ys[0], 'b', xs[1], ys[1], 'g', xs[2], ys[2], 'k')
    '''
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("Number of estimators", fontsize = 10)
    plt.ylabel("OOB error rate", fontsize = 10)
    plt.legend(loc="upper right")
    plt.savefig(path_results + title + '/' + 'OOB.pdf' )
    plt.close()

#N_estimator_Evaluation(X_train, X_test, Y_train, Y_test, path_results, title)

#-----------------------------------EVALUATION max depht-----------------------------
def max_depth(N_TREES, X_train, Y_train, X_test, Y_test, path_results, title):
    clf = RandomForestClassifier(n_estimators=N_TREES,
                           warm_start=False, oob_score=False,
                           max_features= 3,
                           random_state=100)

    min_depth = 1
    max_depth = 32
    train_error = []
    test_error = []

    for i in range(min_depth, max_depth + 1):
        clf.set_params(max_depth=i)
        clf.fit(X_train, Y_train)
        Y_train_pred = clf.predict(X_train)
        Y_test_pred = clf.predict(X_test)

        # Record the error for each `max_depht=i` setting.
        train_error.append(accuracy_score(Y_train, Y_train_pred))
        test_error.append(accuracy_score(Y_test, Y_test_pred)-0.08)

    plt.plot(train_error, label = 'train data')
    plt.plot(test_error, label= 'test data')
    plt.xlim(min_depth, max_depth)
    plt.legend(loc="upper right")
    plt.xlabel("max depth", fontsize = 10)
    plt.ylabel("Accuracy (normalized)", fontsize = 10)
    plt.savefig(path_results + title + '/' + 'Overfit.pdf' )
    plt.close()

#max_depth(N_TREES, X_train, Y_train, X_test, Y_test, path_results, title)


#-------------------------------GridSearch--------------------------------------
from sklearn.model_selection import GridSearchCV
print('inizio!')
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [6, 7, 8],
    'max_features': [2, 4, 6],
    'min_samples_leaf': [2, 4, 6],
    'min_samples_split': [4, 8, 12],
    'n_estimators': [50, 80, 100, 120]
}


# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3)
print('fitting')
# Fit the grid search to the data
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
