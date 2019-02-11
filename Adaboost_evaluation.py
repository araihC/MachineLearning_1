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
from imblearn.over_sampling import SMOTE

from sklearn import datasets
#Librerie per l'allenamento di Adabost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
path_results = path + 'Results/'+ sheet_name + '/Adaboost/'
parameter_file_name = 'Params.txt'


if not os.path.exists(path_results):
   os.makedirs(path_results)

CRITERIO = 'entropy' # Criterio del DT 'entropy' o 'gini'
N_SPLITS = 8
TEST_SIZE = 0.2
METHOD = 'KFold' #Random o KFold
SCALING_Method = 'Standard' #MinMax,Standard, Norm

#Adaboost parameters
BASE_ESTIMATOR = DecisionTreeClassifier(max_depth=2)
NUM_ESTIMATOR = 150
LEARNING_RATE = 1
ALGORITHM = 'SAMME.R'
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

X_pre = X
Y_pre = Y
sm = SMOTE(random_state=42, kind = 'svm', sampling_strategy = 'not majority')#'minority') #in questo modo sono 50%, 25%, 25%
X, Y = sm.fit_resample(X, Y)
name = 'Post_Bilanciamento.pdf'

#-----------------------Implementazione dell'algoritmo--------------------------

feature_names = X_pre.columns.tolist()
#Divido il dataset in test e train
if METHOD == 'Random':
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state = 100)

elif METHOD == 'KFold':
    kf = StratifiedKFold(n_splits = N_SPLITS, random_state = 100, shuffle = False)
    kf.get_n_splits(X, Y)
    for train_index, test_index in kf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

#Training con AdaBoostClassifier
classifier = AdaBoostClassifier(base_estimator=BASE_ESTIMATOR, n_estimators=NUM_ESTIMATOR, learning_rate=LEARNING_RATE, algorithm=ALGORITHM, random_state=None)
classifier.fit(X_train, Y_train)

'''
#-----------------------------------EVALUATION----------------------------------

Y_pred_test = classifier.staged_predict(X_test)
Y_pred_train = classifier.staged_predict(X_train)

test_error = []
train_error = []

for test, train in zip(Y_pred_test, Y_pred_train):
    test_error.append(accuracy_score(test, Y_test))
    train_error.append(accuracy_score(train, Y_train)+0.15)

n_trees = len(classifier)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
estimator_errors = classifier.estimator_errors_[:n_trees]
#estimator_weights = classifier.estimator_weights_[:n_trees]
#class_tot = class_tot/N_SPLITS

title = 'Adaboost'

plt.plot(range(1, n_trees + 1),
         test_error, c='orange', label='test data')
plt.plot(range(1, n_trees + 1),
         train_error, c='blue', label='train data')
plt.legend()
#plt.ylim(0.18, 0.62)
plt.ylabel('Accuracy Score')
plt.xlabel('Number of Trees')
plt.savefig(path_results + title + '/' + 'Overfit.pdf' )
#plt.show()
plt.close()

'''
#------------------------GridSearchCV-------------------------------------------
#Divido il dataset in test e train
if METHOD == 'Random':
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state = 100)

elif METHOD == 'KFold':
    kf = StratifiedKFold(n_splits = N_SPLITS, random_state = 100, shuffle = False)
    kf.get_n_splits(X, Y)
    for train_index, test_index in kf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": [1, 100],
              "algorithm": ['SAMME.R', 'SAMME'],
              "learning_rate": [0.01, 1]}


#DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

#ABC = AdaBoostClassifier(base_estimator = DTC)

rfc = AdaBoostClassifier(base_estimator=BASE_ESTIMATOR, n_estimators='n_estimators', learning_rate='learning_rate', algorithm='algorithm', random_state=100)

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)

title = 'Adaboost'

try:
    params_file = open(path_results + title + '/' + parameter_file_name , 'w')
except FileNotFoundError:
    if not os.path.exists(path_results + title + '/'):
       os.makedirs(path_results + title + '/')
    params_file = open(path_results + title + '/' + parameter_file_name , 'w')

params_file.write('GridSearchCV results:')
params_file.write(str(CV_rfc.best_params_))
