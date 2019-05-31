#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:08:44 2019

@author: thales
"""

#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
    LINK PARA A BIBLIOTECA:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit
'''
#datatrain = pd.read_csv('./testes_outros_algoritmos.csv')

datatrain = pd.read_csv('./zero_um_menos_50_alunos.csv')

datatrain.loc[datatrain['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

X = datatrain_array[:, :14]
y = datatrain_array[:, 14:15]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
precisao = clf.score(X_test,y_test)
print("------Acurácia-------: %f" %(precisao))
#previsao = cross_val_score(clf, X, y)