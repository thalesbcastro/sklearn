#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:58:10 2019

@author: thales
"""

'''
    O pandas é um pacote Python que fornece estruturas de dados rápidas, flexíveis e expressivas, 
    projetadas para tornar o trabalho com dados “relacionais” ou “rotulados” fáceis e intuitivos. 
    O objetivo é ser o bloco de construção fundamental de alto nível para a análise prática de dados 
    do mundo real em Python. Além disso, tem o objetivo mais amplo de se tornar a mais poderosa e 
    flexível ferramenta de análise/manipulação de dados de código aberto disponível em qualquer idioma.
    (https://pandas.pydata.org/pandas-docs/stable/)
'''
import pandas as pd

'''
    NumPy é um pacote em Python para computação científica. Ele contém entre outras coisas:
        1) Um poderoso Objeto Array N-Dimensional;
        2) Funções sofisticadas (broadcasting);
        3) Ferramentas para integrar código C/C++ e Fortran
    (http://www.numpy.org/)
'''
import numpy as np 

'''
    scikit-learn é um ferramenta simples e eficiente para análise e mineração de dados, acessível a todo
    mundo e reutilizável em vários contextos. Construída em NumPy, SciPy e matplotilib. 
    (https://scikit-learn.org/stable/index.html)
    
    model_selection.train_test_split é um módulo que inclui classes e funções para dividir 
    os dados com base em uma estratégia predefinida.
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
'''
from sklearn.model_selection import train_test_split

 
'''
    Lendo o CSV e transformando-o em um DatFrame que é uma estrutura bidimensional, como uma planilha.
'''
#datatrain = pd.read_csv('./variaveis_mysql_com_EA_MLP.csv')
#Arquivo com os estilos calculados segundo o critério de Gustavo
#datatrain = pd.read_csv('./segurança_do_trabalho/Base_de_Dados_IFRN_Seg_Trab_21_01_2019.csv')

#datatrain = pd.read_csv('./comportamentos_alunos_ifrn_definitivo.csv')
datatrain = pd.read_csv('./zero_um_menos_50_alunos.csv')

#datatrain = pd.read_csv('./variaveis_ifrn_seg_trab_14_01_2019.csv')
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


'''
    Vamos acrescentar 4 novos campos no DataFrame para representar os atributos da coluna 
    'estilo_de_aprendizagem'. 
'''
datatrain['y0'] = [1 if x == 0 else 0 for x in datatrain['estilo_de_aprendizagem']]
datatrain['y1'] = [1 if x == 1 else 0 for x in datatrain['estilo_de_aprendizagem']]
datatrain['y2'] = [1 if x == 2 else 0 for x in datatrain['estilo_de_aprendizagem']]
datatrain['y3'] = [1 if x == 3 else 0 for x in datatrain['estilo_de_aprendizagem']]
datatrain['y4'] = [1 if x == 4 else 0 for x in datatrain['estilo_de_aprendizagem']]


'''
    Convertendo o DataFrame para dtype: float64 e retirando o nome das colunas, 
    transformando-o em uma matrix.
'''
datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

'''
    Criando o conjunto de treinamento e teste através da função train_test_split
'''
X = datatrain_array[:,:14]
#y = datatrain_array[:, 15:20]

y = datatrain_array[:, 14:15]
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


'''
    Nesse momento é necessário a padronização dos dados para que a rede neural tenha menos dificuldade
    em convergir. Nesse caso, utilizamos a função StandardScaler proveniente do módulo preprocessing do 
    Scklearn. Essa função, na prática, ignora a forma da distribuição e transforma o dado para forma com 
    média próxima de zero e um desvio padrão próximo a um, ou seja, assume que não temos valores 
    discrepantes nos dados e normaliza tudo (http://minerandodados.com.br/index.php/2017/12/28/pre-processamento-standartization/)
    O score padrão de uma amostra x é calculada como:
    z = (x - u)/s
    Em que u é a média das amostras de treinamento ou zero se o parâmetro with_mean=False e s é o desvio
    padrão das amostras de treinamento ou um se with_std=False. 
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    
'''
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MinMaxScaler()
'''
    Fit apenas para o conjunto de treino pois ela calcula a média e o desvio padrão da distribuição 
    e já “sabe” como irá fazer para padronizar os dados, não sendo necessário fazer o mesmo para os dados de teste
    Diante disso, podemos usar o método transform() para aplicar os cálculos e fazer a transformação nos dados
    (http://minerandodados.com.br/index.php/2017/12/28/pre-processamento-standartization/)
'''
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
    Para criar a MLP é necessário importar a função MLPClassifier do módulo neural_network. 
    (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
'''
from sklearn.neural_network import MLPClassifier

'''
    Aqui começa a criação da arquitetura da MLP
'''
'''
    Previsão e avaliação
'''
from sklearn.metrics import confusion_matrix, accuracy_score

lista_matrix_conf = []
lista_accuracy_calculados = []
i_while = 0

from sklearn.linear_model import LinearRegression, LogisticRegression
'''
    --------------------Teste com regressão linear------------------
'''

mlp = LogisticRegression(solver = 'lbfgs',
                                      max_iter = 1000,
                                      multi_class = 'multinomial')
#mlp = LinearRegression()

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

precisao = mlp.score(X_test,y_test)
print("Precisão: %f" %(precisao))


'''
    -------------------Teste com o LinearSVC--------------------
'''
from sklearn.svm import LinearSVC


'''
    ----------------Teste com o RandomForestClassifier-----------
'''
from sklearn.ensemble import RandomForestClassifier

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

precisao = mlp.score(X_test,y_test)
print("Precisão: %f" %(precisao))





while(i_while <= 49):
    
    '''
        Aqui começa a criação da arquitetura da MLP
    '''
    mlp = MLPClassifier(verbose=True, # Mostrar o erro 
                        tol = 1e-4,
                        hidden_layer_sizes=(20,20),
                        solver='lbfgs',
                        activation ='logistic',
                        learning_rate_init=.001,
                        max_iter=1000)
    #mlp = linear_model.LogisticRegression(solver = 'lbfgs',
    #                                      max_iter = 1000,
    #                                      multi_class = 'multinomial')
    
    #mlp = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #mlp = LinearSVC()
    
    mlp.fit(X_train, y_train)
    
    
    '''
        A função accuracy_score calcula a precisão do subconjunto: o conjunto de rótulos previsto 
        para uma amostra deve corresponder exatamente ao conjunto de rótulos correspondente em y_true
        (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
    '''
#    print("\n")
#    print("-----------Precisão da REDE-----------")
    predictions = mlp.predict(X_test)
#    print("Precisão do subconjunto (accuracy_score): %f" %(accuracy_score(y_test, predictions)))
    lista_accuracy_calculados.append(accuracy_score(y_test, predictions))
    
    '''
        A função score é semelhante a accuracy_score. Ela retorna a precisão média nos
        dados e rótulos de teste fornecidos
    '''
    
    #precisao = mlp.score(X_train,y_train)
    #print("Precisao de treino: %f" %(precisao))
    precisao = mlp.score(X_test,y_test)
#    print("Precisao do subconjunto (score) %f" %(precisao))
#    print("\n")
    
    
    '''
        Para poder calcular a matriz de confusão é necessário transformar o y_test e o predito em listas, pois 
        é o formato de dado que a função confusion_matrix aceita.
    '''
    y_real = []
    y_predito = []
    for i in range(y_test.shape[0]):
        if np.all(y_test[i] == [1, 0, 0, 0, 0]):
            y_real.append(0)
        elif np.all(y_test[i] == [0, 1, 0, 0, 0]):
            y_real.append(1)
        elif np.all(y_test[i] == [0, 0, 1, 0, 0]):
            y_real.append(2)
        elif np.all(y_test[i] == [0, 0, 0, 1, 0]):
            y_real.append(3)
        else:
            y_real.append(4) 
    for i in range(y_test.shape[0]):
        if np.all(predictions[i] == [1, 0, 0, 0, 0]):
            y_predito.append(0)
        elif np.all(predictions[i] == [0, 1, 0, 0, 0]):
            y_predito.append(1)
        elif np.all(predictions[i] == [0, 0, 1, 0, 0]):
            y_predito.append(2)
        elif np.all(predictions[i] == [0, 0, 0, 1, 0]):
            y_predito.append(3)
        else:
            y_predito.append(4) 
    
#    print("-----------Matriz de Confusão-----------")
#    print("Matriz de confusão: %i" %(i_while))
#    print(confusion_matrix(y_real, y_predito))
    matrix_conf = confusion_matrix(y_real, y_predito)
#    print("\n")
    lista_matrix_conf.append((matrix_conf/598))
    i_while = i_while + 1


media_accuracy = sum(lista_accuracy_calculados)/len(lista_accuracy_calculados)
print("Média da precisão: %f" %(media_accuracy), "\n")

media_matrix_confusion = sum(lista_matrix_conf)/len(lista_matrix_conf)
print("Média das matrizes de confusão das 50 iterações: ")
print(np.matrix(media_matrix_confusion))
print("\n")

median_matrix_confusion = (lista_matrix_conf[24] + lista_matrix_conf[25])/2
print("Mediana das matrizes de confusão das 50 iterações: ")
print(np.matrix(median_matrix_confusion))
print("\n")

max_value_accuracy = max(lista_accuracy_calculados)
print("Valor máximo da precisão: %f" %(max_value_accuracy), "\n")

min_value_accuracy = min(lista_accuracy_calculados)
print("Valor mínimo da precisão: %f" %(min_value_accuracy), "\n")

print("----------Parâmetros da rede----------\n")
print("Número de neurônios em cada camada da rede neural")
print(mlp.hidden_layer_sizes)
print("\n")

print("Número máximo de iterações: ")
print(mlp.max_iter)
print("\n")

print("Função para otimização dos pesos: ")
print(mlp.solver)
print(" (is an optimizer in the family of quasi-Newton methods)\n")

print("Função de ativação: ")
print(mlp.activation)
print(" (f(x) = 1 / (1 + exp(-x)))\n")

print("A taxa inicial de aprendizado: ")
print(mlp.learning_rate_init)
print("\n")


