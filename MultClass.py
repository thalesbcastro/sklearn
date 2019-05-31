#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:51:15 2019

@author: thales
"""



import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

#import snips as snp
#snp.prettyplot(matplotlib)

df = pd.read_csv('./testes_outros_algoritmos.csv')
#df = pd.read_csv('./zero_um_menos_50_alunos.csv')
'''
    Número de alunos classificações Ativo, Reflexivo, Pragmático, Teórico e Indefinido da base de dados criada
    
'''

'''
    Código referente ao gráfico de barra das classes
'''
#qtd_estilos = df['estilo_de_aprendizagem'].value_counts()
#qtd_estilos.plot(x='Classes', y='Frequência', 
#                 kind='bar', legend=False, grid=True, figsize=(8, 5))

'''
    Código para plotar os Bloxpots
'''
df['v0'] = df['qtd_de_acessos_pagina'] 
df['v1'] = df['qtd_de_acessos_pasta'] 
df['v2'] = df['qtd_de_acessos_arquivo'] 
df['v3'] = df['qtd_de_acessos_url'] 
df['v4'] = df['numero_de_acesso_por_curso'] 
df['v5'] = df['qtd_mensagens_enviadas'] 
df['v6'] = df['qtd_de_acessos_livro'] 
df['v7'] = df['qtd_acessos_ao_chat'] 
df['v8'] = df['qtd_mensagens_chat'] 
df['v9'] = df['qtd_acessos_wiki'] 
df['v10'] = df['numero_postagens'] 
df['v11'] = df['qtd_de_acessos_tarefa'] 
df['v12'] = df['qtd_de_acessos_forum'] 
df['v13'] = df['qtd_de_acessos_questionario'] 


df_boxplot_1 = pd.DataFrame(df, columns=['v0', 'v1', 'v2','v3', 'v4'])
boxplot_1 = df_boxplot_1.boxplot(column=['v0', 'v1', 'v2','v3', 'v4']) 

df_boxplot_2 = pd.DataFrame(df, columns=['v5', 'v6', 'v7', 'v8', 'v9'])
boxplot_2 = df_boxplot_2.boxplot(column=['v5', 'v6', 'v7', 'v8', 'v9']) 

df_boxplot_3 = pd.DataFrame(df, columns=['v10', 'v11', 'v12', 'v13'])
boxplot_3 = df_boxplot_3.boxplot(column=['v10', 'v11', 'v12', 'v13']) 
#from sklearn.neural_network import MLPClassifier
#
#'''
#    Função custo é o gradiente descendente 
#'''
#
#df.loc[df['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
#df.loc[df['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
#df.loc[df['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
#df.loc[df['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
#df.loc[df['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4
#
#df = df.apply(pd.to_numeric)
#df_array = df.as_matrix()
#
#X = df_array[:,:14]
#y = df_array[:, 14:15]
#y = y.ravel()
#
#'''
#    Procurar saber como funciona isso e o que é
#'''
#'''
#    WARM_START = TRUE: https://stackoverflow.com/questions/35756549/partial-fit-sklearns-mlpclassifier?rq=1
#    https://plot.ly/scikit-learn/plot-mlp-training-curves/
#    https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
#    https://sdsawtelle.github.io/blog/output/week4-andrew-ng-machine-learning-with-python.html
#    https://github.com/fmilepe/avito-contest/blob/master/classifiers.py
#'''
#c = list(zip(X, y))
#X, y = zip(*c)
##X = np.array(X)
##y = np.array(y)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
##old_stdout = sys.stdout
##sys.stdout = mystdout = StringIO()
#
#mlp = MLPClassifier(solver='sgd', verbose=10)
#mlp.hidden_layer_sizes = (15,)
#mlp.tol = 0.000000001
#mlp.learning_rate = 'adaptive'
#mlp.max_iter = 10000 
#mlp.early_stopping = True
#mlp.activation = "logistic"
#mlp.learning_rate_init = 1
#mlp.fit(X_train, y_train)
#y_pred = mlp.predict(X_test)
#print("Precisão da Rede: %f" %(accuracy_score(y_test, y_pred)))
#snp.prettyplot(matplotlib)
#fig, ax = snp.newfig()
#ax.plot(mlp.loss_curve_)
#snp.labs("number of steps", "loss function", "Loss During GD (Rate=0.001)")