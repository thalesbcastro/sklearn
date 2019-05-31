#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:36:37 2019

@author: thales
"""

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


df = pd.read_csv('./testes_outros_algoritmos.csv')
df.loc[df['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
df.loc[df['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
df.loc[df['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
df.loc[df['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
df.loc[df['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

df = df.apply(pd.to_numeric)
df_array = df.as_matrix()

X = df_array[:,:14]
y = df_array[:, 14:15]
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

mlps = []

for label, param in zip(labels, params):
    print("training: %s" % label)
    mlp = MLPClassifier(verbose=0, random_state=0,
                        max_iter=400, **param)
    mlp.fit(X_train, y_train)
    mlps.append(mlp)
    print("Training set score: %f" % mlp.score(X_test, y_test))
    print("Training set loss: %f" % mlp.loss_)
for mlp, label, args in zip(mlps, labels, plot_args):
        plt.plot(mlp.loss_curve_, label=label, **args)


