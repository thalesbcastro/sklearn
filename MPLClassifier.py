#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:08:44 2019

@author: thales
"""

#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
import pandas as pd

'''
    LINK PARA A BIBLIOTECA:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
'''
datatrain = pd.read_csv('./testes_outros_algoritmos.csv')
#datatrain = pd.read_csv('./zero_um_menos_50_alunos.csv')

datatrain.loc[datatrain['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
datatrain.loc[datatrain['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

X = datatrain_array[:, :14]
y = datatrain_array[:, 14:15]
y = MultiLabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = RobustScaler()
##scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15, ), random_state=1)

mlp.fit(X_train, y_train)
precisao = mlp.score(X_test, y_test)
print("------Acurácia-------: %f" %(precisao))