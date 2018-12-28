#Versione 1.0, aggiornata il 28/12/18 - python3.6 - Chiara De Luca
#Questo programma acquisisce in input i file raw e crea un nuovo file xcel con la corretta forma/pulizia
#in particolare:
#   Loading dei dataset a disposizione
#   Normalizzazione del voto di diploma (se mancante, sostituito con la moda)
#   Eliminazione degli immatricolati anno 2918/2019 (non abbiamo info sulla carriera)
#   Normalizzazione dei risultati del Test
#   Anni effettivi di Anni_di_studio
#   CFU normalizzati per anno
#   Sostituisco i valori mancanti di CFU e media con la moda nello stesso croso di studio

#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

path = '/Users/Mac/Desktop/ML/ML_project/Dati_raw/'
save_path = path + '/../' + 'Cleaned_data/'
#loading dei vari file, rendo coerenti gli header

file_name = path + '20142015_14616_PIVC.xlsx'
sheet = '14616CHI_N'
PIVC_1415 = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=6, usecols= 'C:H')
PIVC_1415.rename(columns= {'Pu.Tot.': 'Tot_Test', 'Materia_1': 'Matematica', 'Materia_2': 'Chimica', 'Diploma': 'Voto_Diploma', 'Data nascita': 'Data_nascita'}, inplace = True)
#print(PIVC_1415)

file_name = path + '20152016_14616_PIVC.xls'
sheet = 'Sheet0'
PIVC_1516 = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=7, usecols= 'C:H')
PIVC_1516.rename(columns= {'Pu,Tot,': 'Tot_Test', 'Materia_1': 'Matematica', 'Materia_2': 'Chimica', 'Diploma': 'Voto_Diploma', 'Data nascita': 'Data_nascita'}, inplace = True)
#print(PIVC_1516)

file_name = path + '20162017_SCIENZE_PIVC_Cisia.xlsx'
sheet = 'Foglio1'
PIVC_1617_1 = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=0, usecols= 'B, E:O')
PIVC_1617_1.rename(columns= {'matricola':'Matricola', 'DATA_DI_NASCITA': 'Data_nascita', 'COMUNE_DI_NASCITA':'Comune_nascita', 'CODICE_FISCALE' : 'CF', 'CORSO':'Corso', 'P_MATBASE' : 'Matematica', 'P_BIOL': 'Biologia', 'P_CHIMICA':'Chimica','P_MATEPROBL':'Prob', 'P_SCTERRA': 'Scienze', 'PTEST':'Tot_Test', 'P_FISICA':'Fisica'}, inplace = True)
#print(PIVC_1617_1)

file_name = path + '20162017_SCIENZE_PIVC_Cisia.xlsx'
sheet = 'Foglio2'
PIVC_1617_2 = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=0, usecols= 'K, P:V, Y')
PIVC_1617_2.rename(columns= {'IDENTIFICATIVO_STUDENTE(PREMATRICOLA)':'Matricola', 'DATA_DI_NASCITA': 'Data_nascita', 'COMUNE_DI_NASCITA':'Comune_nascita', 'CODICE_FISCALE' : 'CF', 'GENERE':'Sesso', 'VOTO_DIPLOMA': 'Voto_Diploma', 'TIPO_SCUOLA(COD_CISIA)':'Tipo_Diploma'}, inplace = True)
#print(PIVC_1617_2)

file_name = path + '20172018_SCIENZE_PIVC_Cisia.xlsx'
sheet = 'Foglio1'
PIVC_1718 = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=0, skipfooter=1, usecols= 'B, E:P')
PIVC_1718.rename(columns= {'matricola':'Matricola', 'DATA_DI_NASCITA': 'Data_nascita', 'COMUNE_DI_NASCITA':'Comune_nascita', 'CODICE_FISCALE' : 'CF', 'codice corso':'Corso', 'P_MATBASE' : 'Matematica', 'P_BIOL': 'Biologia', 'P_CHIMICA':'Chimica','P_MATEPROBL':'Prob', 'P_PROBLEMVOLGIN':'Problemi', 'P_SCTERRA': 'Scienze', 'PTEST':'Tot_Test', 'P_FISICA':'Fisica'}, inplace = True)
#print(PIVC_1718)

file_name = path + 'SANTANASTASIO_STATISTICHE_SSMMFFNN_111018.xlsx'
sheet = 'Foglio1'
CARRIERA = pd.read_excel(io=file_name, sheet_name=sheet, skiprows=0, skipfooter=1, usecols= 'A:Q')
CARRIERA.rename(columns= {'MATRICOLA':'Matricola', 'ANNO_ACCADEMICO_IMMATRICOLAZIONE':'Anno_Immatricolazione', 'CODICE_CORSO':'Corso', 'ANNO_ACCADEMICO_ULTIMA_ISCRIZIONE':'Anno_Iscrizione', 'STATO_LAUREA':'Stato_Laurea', 'PUNTEGGIO_TEST_DI_INGRESSO':'Tot_Test', 'VOTO_LAUREA':'Voto_Laurea', 'SESSO':'Sesso', 'ANNO_NASCITA': 'Data_nascita', 'TIPOLOGIA_DIPLOMA':'Tipo_Diploma', 'VOTO_DIPLOMA': 'Voto_Diploma', 'CFU_SOSTENUTI':'CFU', 'MEDIA_DEI_VOTI_PESATA_PER_I_CFU':'Media_Pesata'}, inplace = True)

#================ Modifico ora il contenuto dei singoli DataFrame. ==============
#================================================================================

#--------------> Sostituisco le date di nascita con gli anni

PIVC_1516.Data_nascita = pd.to_datetime(PIVC_1516.Data_nascita)
PIVC_1718.Data_nascita = pd.to_datetime(PIVC_1718.Data_nascita)

PIVC_1415.Data_nascita=PIVC_1415.Data_nascita.dt.year
PIVC_1516.Data_nascita=PIVC_1516.Data_nascita.dt.year
PIVC_1617_1.Data_nascita=PIVC_1617_1.Data_nascita.dt.year
PIVC_1617_2.Data_nascita=PIVC_1617_2.Data_nascita.dt.year
PIVC_1718.Data_nascita=PIVC_1718.Data_nascita.dt.year

#--------------> Riporto ora tutti i voti di maturità in centesimi

for i in PIVC_1617_2.BASE_VOTO_DIPLOMA: #qui 'riempio' i dati mancanti nella base del diploma esprimento il punteggio in 100esimi
    if np.isnan(i):
        i = 100

PIVC_1617_2.Voto_Diploma = 100*PIVC_1617_2.Voto_Diploma.div(PIVC_1617_2.BASE_VOTO_DIPLOMA.values,axis=0) #Calcolo la proporzione
PIVC_1617_2.drop(labels='BASE_VOTO_DIPLOMA', axis=1, inplace = True) #elmino la colonna di punteggi di riferimento

mode = PIVC_1617_2.Voto_Diploma.mode()  #sostituisco ai valori mancanti la moda del set
for i in PIVC_1617_2.Voto_Diploma:
    if np.isnan(i) or i == 0 or i < 60:
        i = mode

mode = PIVC_1415.Voto_Diploma.mode()  #sostituisco ai valori mancanti la moda del set
for i in PIVC_1415.Voto_Diploma:
    if np.isnan(i) or i == 0 or i < 60:
        i = mode

mode = PIVC_1516.Voto_Diploma.mode()  #sostituisco ai valori mancanti la moda del set
for i in PIVC_1516.Voto_Diploma:
    if np.isnan(i) or i == 0 or i < 60:
        i = mode


#Ora modifico il file Carriera

for i in CARRIERA.VOTO_MASSIMO_DIPLOMA: #qui 'riempio' i dati mancanti nella base del diploma esprimento il punteggio in 100esimi
    if np.isnan(i):
        i = 100

CARRIERA.Voto_Diploma = 100*CARRIERA.Voto_Diploma.div(CARRIERA.VOTO_MASSIMO_DIPLOMA.values,axis=0) #Calcolo la proporzione
CARRIERA.drop(labels='VOTO_MASSIMO_DIPLOMA', axis=1, inplace = True) #elmino la colonna di punteggi di riferimento

mode = CARRIERA.Voto_Diploma.mode()  #sostituisco ai valori mancanti la moda del voto diploma
for i in CARRIERA.Voto_Diploma:
    if np.isnan(i) or i == 0 or i < 60:
        i = mode

CARRIERA = CARRIERA[CARRIERA.Anno_Immatricolazione != 2019]     #elimino qui tutti quelli iscritti per il 2018/2019 non avendo noi i dati

#--------------> Test di ingresso

#devo omogenizzare ora il risultato del test di ingresso nel dataset CARRIERA
#suppongo che almeno una persona abbia preso il massimo in ciascun anno, normalizzo il punteggio in 30esimi

VAR_NORM = 30 #punteggio espresso in termini di VAR_NORM
temp_2014 = []
temp_2015 = []
temp_2016 = []
temp_2017 = []
temp_2018 = []

CARRIERA.Tot_Test = CARRIERA.Tot_Test.astype(float)
CARRIERA.Anno_Immatricolazione = CARRIERA.Anno_Immatricolazione.astype(int)

for i in range(0, CARRIERA.shape[0]):
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2014 and CARRIERA.iloc[i]['Tot_Test'] != -99:
        temp_2014.append(CARRIERA.iloc[i]['Tot_Test'])
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2015 and CARRIERA.iloc[i]['Tot_Test'] != -99:
        temp_2015.append(CARRIERA.iloc[i]['Tot_Test'])
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2016 and CARRIERA.iloc[i]['Tot_Test'] != -99:
        temp_2016.append(CARRIERA.iloc[i]['Tot_Test'])
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2017 and CARRIERA.iloc[i]['Tot_Test'] != -99:
        temp_2017.append(CARRIERA.iloc[i]['Tot_Test'])
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2018 and CARRIERA.iloc[i]['Tot_Test'] != -99:
        temp_2018.append(CARRIERA.iloc[i]['Tot_Test'])

for i in range(0, CARRIERA.shape[0]):

    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2014:
        if CARRIERA.iloc[i]['Tot_Test'] == -99:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = np.mean(temp_2014)*VAR_NORM/max(temp_2014)
        else:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')]*VAR_NORM/max(temp_2014)
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2015:
        if CARRIERA.iloc[i]['Tot_Test'] == -99:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = np.mean(temp_2015)*VAR_NORM/max(temp_2015)
        else:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')]*VAR_NORM/max(temp_2015)
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2016:
        if CARRIERA.iloc[i]['Tot_Test'] == -99:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = np.mean(temp_2016)*VAR_NORM/max(temp_2016)
        else:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')]*VAR_NORM/max(temp_2016)
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2017:
        if CARRIERA.iloc[i]['Tot_Test'] == -99:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = np.mean(temp_2017)*VAR_NORM/max(temp_2017)
        else:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')]*VAR_NORM/max(temp_2017)
    if CARRIERA.iloc[i]['Anno_Immatricolazione'] == 2018:
        if CARRIERA.iloc[i]['Tot_Test'] == -99:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = np.mean(temp_2018)*VAR_NORM/max(temp_2018)
        else:
            CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tot_Test')]*VAR_NORM/max(temp_2018)

del(temp_2014, temp_2015, temp_2016, temp_2017, temp_2018)

#--------------> Tipologia di diploma

#Faccio l'elenco delle varie tipologie di diploma
#sostituisco ai valori mancanti la moda
for i in range(0, CARRIERA.shape[0]):
    if CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tipo_Diploma')] == '-99':
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('Tipo_Diploma')] = CARRIERA.Tipo_Diploma.mode()[0]

#--------------> Regione di diploma

#Se -99 scrivo ESTERO
for i in range(0, CARRIERA.shape[0]):
    #print(CARRIERA.REGIONE_ISTITUTO_DIPLOMA)
    if CARRIERA.iloc[i, CARRIERA.columns.get_loc('REGIONE_ISTITUTO_DIPLOMA')] == '-99':
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('REGIONE_ISTITUTO_DIPLOMA')] = 'ESTERO'

#--------------> CFU e MEDIA DEI CFU

#Creo una nuova colonna con 'numero di anni'
CARRIERA.Anno_Iscrizione = CARRIERA.Anno_Iscrizione.astype(int)

temp_anni = []
for i in range(0, CARRIERA.shape[0]):
    temp_anni.append(CARRIERA.iloc[i]['Anno_Iscrizione'] - CARRIERA.iloc[i]['Anno_Immatricolazione'])

CARRIERA.loc[:,'Anni_di_studio'] = temp_anni
del(temp_anni)

#Normalizzo i CFU a CFU/ANNO
for i in range(0, CARRIERA.shape[0]):
    if CARRIERA.iloc[i, CARRIERA.columns.get_loc('Anni_di_studio')] != 0:
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')] = CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')]/CARRIERA.iloc[i, CARRIERA.columns.get_loc('Anni_di_studio')]
    elif CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')] < 0:
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')] = 0

#Creo elenco delle media e cfu/anno per ciascun corso di studio
elenco_corsi = []
cfu_corso = []
media_corso = []
for corso in CARRIERA.Corso:
    if not corso in elenco_corsi: #se ancora non è stato analizzato
        somma_cfu = 0
        somma_media = 0
        count_cfu = 0
        count_media = 0
        for i in range(0, CARRIERA.shape[0]):
            if CARRIERA.iloc[i]['Corso'] == corso:
                if CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')] > 0:
                    somma_cfu += CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')]
                    count_cfu +=1
                if CARRIERA.iloc[i, CARRIERA.columns.get_loc('Media_Pesata')] > 18:
                    somma_media += CARRIERA.iloc[i, CARRIERA.columns.get_loc('Media_Pesata')]
                    count_media += 1
        cfu_corso.append(somma_cfu/count_cfu)
        media_corso.append(somma_media/count_media)
        elenco_corsi.append(corso)

del somma_media, count_media, somma_cfu, count_cfu
#NB:sia la media che i cfu dipendono fortemente dal corso seguito!

#Sostituisco alla media la media delle medie per lo stesso corso ove mancante, e lo stesos per i cfu
for i in range(0, CARRIERA.shape[0]):
    if CARRIERA.iloc[i]['Media_Pesata'] < 18:
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('Media_Pesata')] = media_corso[elenco_corsi.index(CARRIERA.iloc[i]['Corso'])]
    if CARRIERA.iloc[i]['CFU'] < 0:
        CARRIERA.iloc[i, CARRIERA.columns.get_loc('CFU')] = cfu_corso[elenco_corsi.index(CARRIERA.iloc[i]['Corso'])]

#inserisco una nuova colonna che classifica la media in una 'fascia'
#A: 28-30
#B: 26-28
#C: 23-26
#D: 18-23
zona_media = []
for i in range(0, CARRIERA.shape[0]):
    if CARRIERA.iloc[i]['Media_Pesata'] >= 18 and CARRIERA.iloc[i]['Media_Pesata'] < 23:
        zona_media.append('D')
    if CARRIERA.iloc[i]['Media_Pesata'] >= 23 and CARRIERA.iloc[i]['Media_Pesata'] < 26:
        zona_media.append('C')
    if CARRIERA.iloc[i]['Media_Pesata'] >= 26 and CARRIERA.iloc[i]['Media_Pesata'] < 28:
        zona_media.append('B')
    if CARRIERA.iloc[i]['Media_Pesata'] >= 28 and CARRIERA.iloc[i]['Media_Pesata'] <= 30:
        zona_media.append('A')
CARRIERA.loc[:,'Media_Zona'] = zona_media
del(zona_media)
#--------------------------------Scrivo il dataset opportunemente modificato su un nuovo file.....................

CARRIERA.drop(labels='CF', axis=1, inplace = True) #elmino la colonna del CF
CARRIERA.drop(labels='Matricola', axis=1, inplace = True) #elmino la colonna di matricole
CARRIERA.drop(labels='Anno_Iscrizione', axis=1, inplace = True) #elmino la colonna dell'anno di iscrizione
CARRIERA.drop(labels='Anno_Immatricolazione', axis=1, inplace = True) #elmino la colonna dell'immatricolazione (mi interessa solo la differenza)
CARRIERA.drop(labels='PROVINCIA_ISTITUTO_DIPLOMA', axis=1, inplace = True) #elmino la colonna dell'immatricolazione (mi interessa solo la differenza)
CARRIERA.drop(labels='Stato_Laurea', axis=1, inplace = True) #elmino la colonna dell'immatricolazione (mi interessa solo la differenza)
CARRIERA.drop(labels='Voto_Laurea', axis=1, inplace = True) #elmino la colonna dell'immatricolazione (mi interessa solo la differenza)

CARRIERA.to_excel(save_path + 'CARRIERA.xlsx', 'Minimal')
