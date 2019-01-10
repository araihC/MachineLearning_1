#Versione 1.0, aggiornata il 29/12/18 - python3.6 - Chiara De Luca
#Questo programma è una prima semplica implementazione con il metodo di SVM
#Date le informazioni di background du uno studente si vuole indentificare in quale fascia
#risiede la sua media.
#Volendo un algoritmo in grado di classificare i risultati utilizziamo l'algoritmo

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

from sklearn import datasets
#Librerie per l'allenamento DT
from sklearn.model_selection import train_test_split
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

path = '/Users/Mac/Desktop/ML/ML_project/'

path_data = path + 'Cleaned_data/'
file_name = path_data + 'CARRIERA.xlsx'
sheet_name = 'Minimal_4'
evaluation_file_name = 'Evaluation.txt'

path_results = path + 'Results/'+ sheet_name + '/DT/'
if not os.path.exists(path_results):
   os.makedirs(path_results)


TEST_SIZE = 0.2
METHOD = 'KFold' #Random o KFold
#SCALING_Method = 'Standard' #MinMax,Standard, Norm
CRITERIO = 'entropy' # Criterio del DT 'entropy' o 'gini'
MAX_DEPTH = 5
N_SPLITS = 8

#===============================================================================

def f_importances(coef, names):
    imp = coef[0]
    #imp, names = zip(imp,names)
    #imp, names = zip(*sorted(zip(imp,names)))

    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.title('Feature importance')
    #plt.show()
    plt.savefig(path_results + 'Feature_importance' + '.pdf')
    plt.close()

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

#Valuatiamo ora se sono bilanciate o meno le classi
def plot_pie(y, path_results, name):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct=make_autopct(sizes))
    ax.axis('equal')
    plt.savefig(path_results + name)
    plt.close()

name = 'Pre_Bilanciamento.pdf'
plot_pie(Y, path_results, name)

#Ok, ora facciamo in modo di renderle bilanciate
'''
sampling_strategy = 'not majority'

rus = RandomOverSampler(sampling_strategy=sampling_strategy)
X_pre = X
Y_pre = Y
X, Y = rus.fit_resample(X, Y)

sampling_strategy = 'not minority'
tl = TomekLinks(sampling_strategy)
X_pre = X
Y_pre = Y
X, Y = tl.fit_resample(X, Y)
'''

X_pre = X
Y_pre = Y
sm = SMOTE(random_state=42, kind = 'svm', sampling_strategy = 'not majority')#'minority') #in questo modo sono 50%, 25%, 25%
X, Y = sm.fit_resample(X, Y)

name = 'Post_Bilanciamento.pdf'
plot_pie(Y, path_results, name)

#-----------------------Implementazione dell'algoritmo--------------------------

#Divido il dataset in test e train
if METHOD == 'Random':
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state = 100)

elif METHOD == 'KFold':
    kf = StratifiedKFold(n_splits = N_SPLITS, random_state = 100, shuffle = False)
    kf.get_n_splits(X,Y)
    for train_index, test_index in kf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
'''
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
'''


# we create an instance of SVM and fit out data.
#alleno il classifier
classifier = DecisionTreeClassifier(class_weight = None, criterion = CRITERIO, max_depth = MAX_DEPTH, random_state=100, splitter='best')
classifier.fit(X_train, Y_train)

#-----------------------------------EVALUATION----------------------------------
title = 'DT'

try:
    evaluation_file = open(path_results + title + '/' + evaluation_file_name , 'w')
except FileNotFoundError:
    if not os.path.exists(path_results + title + '/'):
       os.makedirs(path_results + title + '/')
    evaluation_file = open(path_results + title + '/' + evaluation_file_name , 'w')

Y_pred = classifier.predict(X_test)
print(Y_pred)
print(Y_test)
#evaluate train set error
#train_error = classifier.score(X_train, Y_train)
#test_error = classifier.score(X_test, Y_test)
#print(train_error, test_error)
#-------------> EVALUATION

evaluation_file.write('DT_EVALUATION:')
evaluation_file.write('\n\nConfusion_matrix:\n')
evaluation_file.write(str(confusion_matrix(Y_test, Y_pred)))
evaluation_file.write('\n\nClassification report:\n')
evaluation_file.write(str(classification_report(Y_test, Y_pred)))
#evaluation_file.write("\nAccuracy is "+str(accuracy_score(Y_test,Y_pred)*100) + '\n')
evaluation_file.write('\n\n')
cm = ConfusionMatrix(Y_test, Y_pred)
evaluation_file.write(str(cm.stats()))

sn.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, cmap='YlGnBu')
plt.savefig(path_results + title + '/' + 'Confusion_Matrix.pdf' )
#plt.show()
plt.close()

print("\nAccuracy is "+str(accuracy_score(Y_test,Y_pred)*100) + '\n')

#X_test_plot = scaling.inverse_transform(X_test)
value = 1.5
width = 100

plot_decision_regions(X=X_test, y=Y_test, clf=classifier, legend=2,
              feature_index=[1, 2],                        #these one will be plotted
              filler_feature_values={0: value, 3:value, 4:value, 5:value},  #these will be ignored
              filler_feature_ranges={0: width, 3: width, 4:width, 5:width})

plt.xlabel('Voto del test')
plt.ylabel('Voto del diploma')
plt.title(title)
#plt.show()
plt.savefig(path_results + title + '/' + 'Contours.pdf' )
plt.close()
