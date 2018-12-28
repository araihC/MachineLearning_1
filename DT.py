#Versione 1.0, aggiornata il 28/12/18 - python3.6 - Chiara De Luca
#Questo programma è una prima semplica implementazione con il metodo di DT
#Date le informazioni di background du uno studente si vuole indentificare in quale fascia
#risiede la sua media.
#Volendo un algoritmo in grado di classificare i risultati utilizziamo l'algoritmo Decision Tree Classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Librerie per l'allenamento DT
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Librerie per l'evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#Librerie per il preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Librerire per il plot
import pydot
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

#----------------------------- Parameter definition ----------------------------
path = '/Users/Mac/Desktop/ML/ML_project/'
path_data = path + 'Cleaned_data/'
path_results = path + 'Results/DT/'

file_name = path_data + 'CARRIERA.xlsx'
sheet_name = 'Minimal'

TEST_SIZE = 0.30 #Percentuale di dati utilizzati per il test
CRITERIO = 'entropy' # Criterio del DT 'entropy' o 'gini'
MAX_DEPTH = 4

#===============================================================================

#Loading del dataset
dataset = pd.read_excel(io=file_name, sheet_name=sheet_name)

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

#-------------> ALLENAMENTO

#Divido il dataset in test e train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state = 100)
#alleno il classifier
classifier = DecisionTreeClassifier(class_weight = None, criterion = CRITERIO, max_depth = MAX_DEPTH, random_state=100, splitter='best')

classifier.fit(X_train, Y_train)
#faccio ora delle predizioni su i test data
Y_pred = classifier.predict(X_test)

#-------------> EVALUATION
evaluation_file = open(path_results+'Evaluation.txt', 'w')
evaluation_file.write('DT_EVALUATION: \n\nTEST_SIZE = ' + str(TEST_SIZE) + '\nMAX_DEPTH = ' + str(MAX_DEPTH) + '\nCRITERIO = ' + CRITERIO)
evaluation_file.write('\n\nConfusion_matrix:\n')
evaluation_file.write(str(confusion_matrix(Y_test, Y_pred)))
evaluation_file.write('\n\nClassification report:\n')
evaluation_file.write(str(classification_report(Y_test, Y_pred)))
evaluation_file.write("\nAccuracy is "+str(accuracy_score(Y_test,Y_pred)*100))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print ("Accuracy is ", accuracy_score(Y_test,Y_pred)*100)

#-------------> DT printing

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf(path_results + 'DT.pdf')
