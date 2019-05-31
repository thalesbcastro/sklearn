#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:52:53 2019

@author: thales
"""

# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
# load dataset


'''
    https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
'''

def validacao_cruzada(X, y):
    kf = KFold(len(y), n_folds=2)
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
    return X_train, X_test, y_train, y_test

def treino_predicao_arquitetura(modelos, x1, x2, y1, y2):
    for nome, arq in modelos:
        arq.fit(x1, y1)
        score_treino = arq.score(x1, y1)
        y_pred = arq.predict(x2)
        score_previsao = accuracy_score(y2, y_pred)
        print('%s apresentou:\nEscore de treino de %f \nEscore de previsão de %f'
              %(nome, score_treino, score_previsao))
        print("Matriz de confusão:")
        print(confusion_matrix(y2, y_pred))

'''
    Lendo o arquivo, transformando as classes em valores númericos e em matriz, logo em seguida.
'''
dados = pd.read_csv('./testes_outros_algoritmos.csv')
dados.loc[dados['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
dados.loc[dados['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
dados.loc[dados['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
dados.loc[dados['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
dados.loc[dados['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4

array = dados.values
X = array[:, :14]
y = array[:, 14:15]
y = y.ravel()

'''
   Modelos criados 
'''
mlp = MLPClassifier(
          verbose=0, random_state=1, max_iter=200,
          hidden_layer_sizes=(100, 250), solver='adam', learning_rate_init=0.01
        )
svc = SVC()
gsnb = GaussianNB()
dtc = DecisionTreeClassifier()
kNbC = KNeighborsClassifier()
lgrCV = LogisticRegressionCV()
lgr = LogisticRegression()
modelos = []

modelos.append(('MLP', mlp))
modelos.append(('SVM', svc))
modelos.append(('Gaussian', gsnb))
modelos.append(('DTClass', dtc))
modelos.append(('KNN', kNbC))
modelos.append(('LRCV', lgrCV))
modelos.append(('LR', lgr))

'''
    Chamo a função que usa a validação cruzada para melhor escolher os dados de treinamento
'''
'''
    Ver uma função que trate os dados antes de fazer a validação cruzada.
'''
X_train, X_test, y_train, y_test = validacao_cruzada(X, y)

treino_predicao_arquitetura(modelos, X_train, X_test, y_train, y_test)





# evaluate each model in turn
#resultados = []
#nomes_modelos = []
##metrica = 'accuracy'
#for nome, modelo in modelos:
##	kfold = model_selection.KFold(n_splits=10, random_state=7)
##	resultados_cv = model_selection.cross_val_score(modelo, X, Y, cv=kfold, scoring=metrica)
##	resultados.append(resultados_cv)
#	nomes_modelos.append(nome)
#	msg = "%s: %f (%f)" % (nome, resultados_cv.mean(), resultados_cv.std())
#    y_pred = predict(X_test)
#    score = accuracy_score(y_real, y_pred)
#    msg = "%s: %f (%f)\n" % (nome, score)

#	print(msg)
#    print(matrix_confusion(y_real, y_pred))
    
# boxplot algorithm comparison
#fig = plt.figure()
#fig.suptitle('Algoritmos Multiclasses Sklearn')
#ax = fig.add_subplot(111)
#plt.boxplot(resultados)
#ax.set_xticklabels(nomes_modelos)
#plt.show()