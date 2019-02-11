
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sn

#Librerie per il sampling (imbalanced) dei dati
from imblearn.datasets import make_imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

#Librerie per l'allenamento SVM
from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

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
path = '/Users/Mac/Desktop/ML/ML_project/'

path_data = path + 'Cleaned_data/'
file_name = path_data + 'CARRIERA.xlsx'
sheet_name = 'Minimal_4'
evaluation_file_name = 'Evaluation.txt'

path_results = path + 'Results/'+ sheet_name + '/SVM/'
if not os.path.exists(path_results):
   os.makedirs(path_results)


TEST_SIZE = 0.2
N_SPLITS = 6

C = 1.0  # SVM regularization parameter
POLY_DEGREE = 2 #3
GAMMA = 0.2 #0.7

TEST_SIZE = 0.2
N_SPLITS = 6
METHOD = 'KFold' #Random o KFold
SCALING_Method = 'Standard' #MinMax,Standard, Norm
C = 1.0  # SVM regularization parameter
POLY_DEGREE = 3 #3
GAMMA = 'scale'#0.2 #0.7
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

X_pre = X
Y_pre = Y
sm = SMOTE(random_state=42, kind = 'svm', sampling_strategy = 'not majority')#'minority') #in questo modo sono 50%, 25%, 25%
X, Y = sm.fit_resample(X, Y)

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

#Data scaling
def SCALING_DATA(SCALING, X_train, X_test):
    if SCALING == 'MinMax':
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)

    elif SCALING == 'Standard':
        scaling = StandardScaler().fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)

    elif SCALING == 'Norm':
        scaling = Normalizer().fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)

    elif SCALING == 'Uniform':
        scaling = QuantileTransformer()
        X_train = scaling.fit_transform(X_train)
        X_test = scaling.transform(X_test)

    return X_train, X_test, scaling

X_train, X_test, scaling = SCALING_DATA(SCALING_Method, X_train, X_test)

#---------------------------GridSearch------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'C':[1.0, 10, 0.5]}
svr = svm.SVC(kernel='linear', max_iter = 100000000)
clf = GridSearchCV(svr, parameters, cv = 3)
clf.fit(X_train,Y_train)
print(clf.best_params_)

parameters = {'C':[1.0, 10, 0.5], 'degree':[2,3,4], 'gamma':[0.1,0.2,0.3,0.4]}
svr = svm.SVC(kernel='rbf', max_iter = 100000000)
clf = GridSearchCV(svr, parameters, cv = 3)
clf.fit(X_train,Y_train)
print(clf.best_params_)

parameters = {'C':[1.0, 10, 0.5], 'degree':[2,3,4], 'gamma':[0.1,0.2,0.3,0.4]}
svr = svm.SVC(kernel='poly', max_iter = 100000000)
clf = GridSearchCV(svr, parameters, cv = 3)
clf.fit(X_train,Y_train)
print(clf.best_params_)

#{'C': 10}
#{'C': 10, 'degree': 2, 'gamma': 0.4}
