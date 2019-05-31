#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:10:54 2018

@author: thales
"""

import pandas as pd 
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


todasTurmas = pd.read_excel("TodasTurmasTeste.xlsx")
comportamentosAlunos = pd.read_excel("Base de dados com alunos de 2017 e 2018 e seus estilos - nomes concatenados.xlsx")
X = np.array(todasTurmas.iloc[:, 50:54].values)

kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X)


'''
    -- VERIFICAR O K IDEAL --
    O método de Elbow mostra o somatório da variância dos dados 
    em relação ao número de clusters para verificar até que ponto 
    com o aumento do número de clusters não existe ganho.
    VÊ SE GUSTAVO JÁ OUVIU FALAR NESSE MÉTODO 
'''
#wcss = []
#for i in range(1, 11):
#    kmeans = KMeans(n_clusters = i, init = 'random')
#    kmeans.fit(X)
#    print (i, kmeans.inertia_)
#    wcss.append(kmeans.inertia_) # Valor do erro quadrático 
#plt.plot(range(1, 11), wcss)
#plt.title('Metodo de Elbow')
#plt.xlabel('Numero de clusters')
#plt.ylabel('WSS')
#plt.show()

'''
    http://minerandodados.com.br/index.php/2018/02/02/algoritmo-k-means-python-passo-passo/
    
'''
#kmeans.labels_ o cluster que a instância de dados foi atribuído
#kmeans.cluster_centers_ são os centroides

#plt.scatter(X[:, 0], X[:, 1], s = 100, c = kmeans.labels_)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'red', label = 'Centroids')
#plt.xlabel('Ativo')
#plt.ylabel('Reflexivo')
#plt.show()
#
#plt.scatter(X[:, 2], X[:, 3], s = 100, c = kmeans.labels_)
#plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 300, c = 'red', label = 'Centroids')
#plt.xlabel('Teorico')
#plt.ylabel('Pragmatico')
#plt.show()

df = todasTurmas.iloc[:, 50:54]
df['k-Grupos'] = kmeans.labels_

#sb.pairplot(df, hue='k-Grupos')

df['Alunos'] = todasTurmas.iloc[:, 0].values
df_1 = comportamentosAlunos.iloc[:, 3:9]

#junção dos dataframes pelo campo Alunos
df_merge = pd.merge(df, df_1, on=['Alunos'], how='outer')
#retirando do dataframe NaN
df_completo = df_merge.dropna()

#Transformando em xlsx 
#writer = pd.ExcelWriter('arquivo_completo.xlsx')
#df_completo.to_excel(writer, 'sheet1')
#writer.save()





#grupo = []
#for i in range(1254):
#    data_predict = np.reshape(X[i, 0:4], (1, -1))
#    grupo.append(kmeans.predict(data_predict)[0])
#    
#    
#dataFrameNovo = pd.DataFrame({'Nome Completo': todasTurmas.iloc[:, 0].values,
#                              'Ativo': X[:, 0],
#                              'Reflexivo': X[:, 1],
#                              'Teorico': X[:, 2],
#                              'Pragmatico': X[:, 3],
#                              'EA': grupo})