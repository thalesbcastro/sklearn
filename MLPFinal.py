#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:37:39 2019

@author: thales
"""

#'solver': 'adam', 'learning_rate_init': 0.01


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, precision_score

import pandas as pd

df = pd.read_csv('./testes_outros_algoritmos.csv')
df.loc[df['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
df.loc[df['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
df.loc[df['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
df.loc[df['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
df.loc[df['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

df = df.apply(pd.to_numeric)
df_array = df.as_matrix()


X = df_array[:, :14]
y = df_array[:, 14:15]
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)

X_train = MinMaxScaler().fit_transform(X_train)

mlp = MLPClassifier(solver='adam', learning_rate_init=0.01,
                    max_iter=400, verbose=0, random_state=0)
mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Training set loss: %f" % mlp.loss_)

y_pred = mlp.predict(X_test)

'''
    https://scikit-learn.org/stable/modules/model_evaluation.html
'''

print("scccuracy_score predicao: %f" % accuracy_score(y_test, y_pred))
matriz_confusao = confusion_matrix(y_test, y_pred)

print('mean_absolute_error: %f: ' % mean_absolute_error(y_test, y_pred))
print('precision_score macro: %f ' % precision_score(y_test, y_pred, average='macro'))
print('precision_score micro: %f ' % precision_score(y_test, y_pred, average='micro'))