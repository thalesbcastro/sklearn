#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:16:57 2019

@author: thales
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression

'''
    LINK PARA A BIBLIOTECA: 
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
'''


#datatrain = pd.read_csv('./testes_outros_algoritmos.csv')
datatrain = pd.read_csv('./zero_um_menos_50_alunos.csv')

'''
    Contando os valores da coluna estilo_de_aprendizagem em número e porcentagem.
'''
print("------------Quantidade de cada EA-----------------")
print(datatrain['estilo_de_aprendizagem'].value_counts())
print("\n")
print("------------Porcentagem de cada EA-----------------")
print(datatrain['estilo_de_aprendizagem'].value_counts(normalize=True))
print("\n")
'''
    O atributo .loc é utilizado para se selecionar pelo índice ou rótulo. Aqui ele é utilizado para transformar
    em valores numéricos a coluna 'estilo_de_aprendizagem'
'''
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

'''
    Criando o conjunto de treinamento e teste através da função train_test_split
'''
X = datatrain_array[:,:14]

y = datatrain_array[:, 14:15]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
    --------------------Teste com regressão logística Multiclasse------------------
'''
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)
#mlp = LinearRegression()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

precisao = clf.score(X_test,y_test)
print("---------Precisão---------: %f" %(precisao))