"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.
"""

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.cross_validation import KFold
#from sklearn.model_selection import train_test_split

params = [{'solver':'sgd', 'activation':'tanh', 'learning_rate':'adaptive',  'momentum':0.9,
           'nesterovs_momentum': False, 'early_stopping': True},
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
          {'solver': 'adam', 'learning_rate_init': 0.01, 'beta_1': 0.95, 
           'beta_2': 0.99999}]

labels = ["Estratégia 1", "Estratégia 2", "Estratégia 3",
          "Estratégia 4","Estratégia 5", "Estratégia 6",
          "Estratégia 7"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '--'}]

from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, jaccard_similarity_score, mean_squared_error
import pandas as pd
mlps = []
matriz_confusao = []
def plot_on_dataset(X_train, X_test, y_train, y_test):
    # for each dataset, plot learning for each learning strategy
    print("\nTreinando dataset Moodle")
    ax.set_title("")
    
    
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    

    '''
        Cria a arquitetura MLPClassifier para diferentes parêmetros de learning_rate,
        momentum, nesterovs_momentum=True or False e learning_rate_init. Treinando a rede
        com max_iter=200 e acrescentando dentro de uma lista cada MLP treinada com os respectivos
        parâmetros. Mostra o score e o loss_ pra cada uma.
    '''
    for label, param in zip(labels, params):
        print("Treinando com: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0, hidden_layer_sizes=(100, 260),
                            max_iter=100000, **param)
        mlp.fit(X_train, y_train)
        mlps.append(mlp)
        print("Score de treinamento: %f" % mlp.score(X_train, y_train))
        print("loss de treinamento: %f" % mlp.loss_)
        
        print('-------Precisão de predição para %s: ' %(label) )
        y_pred = mlp.predict(X_test)
        matriz_confusao.append(confusion_matrix(y_test, y_pred))
        
#        print(confusion_matrix(y_test, y_pred))
        #Olhar a função accuracy_score e saber quando é um resultado bom ou ruim
#        print('Acerto na previsão (mpl.score): %f' % mlp.score(y_test, y_pred))
        print(precision_score(y_test, y_pred, average=None))
        print('Acerto na previsão (accuracy_score): %f' % accuracy_score(y_test, y_pred))
        print('Erro quadrático médio: %f' % mean_squared_error(y_test, y_pred))
        print('Índice de Jaccard (multiclass): %f' % jaccard_similarity_score(y_test, y_pred))
    '''
        Cria os gráficos para loss_curve_ para rede com os respectivos parâmetros. Por exemplo, a loss_curve_ foi
        criada para a mlps[1] para solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0, 'learning_rate_init': 0.2,
        e por aí vai
    '''
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)




fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(12)
# load / generate some toy datasets
    
df = pd.read_csv('./testes_outros_algoritmos.csv')
#df = pd.read_csv('./zero_um_menos_50_alunos.csv')
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

X2 = normalize(X)
'''
    Usando validação cruzada 
'''
kf = KFold(len(y), n_folds=2)
i = 0
X_train, X_test, y_train, y_test = [], [], [], []
for train, test in kf:
	i = i + 1
	print("Treinamento", i)

		# dividindo dataset em treino e test
		#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1)
	X_train, X_test, y_train, y_test = X2[train], X2[test], y[train], y[test]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
#                                                    random_state=0)

data_sets = [(X_train, X_test, y_train, y_test)]

for data in data_sets:
    plot_on_dataset(*data)

'''
    Para poder visualizar melhor cada Matriz de confusão
'''
mt_1 = pd.DataFrame(matriz_confusao[0])
mt_2 = pd.DataFrame(matriz_confusao[1])
mt_3 = pd.DataFrame(matriz_confusao[2])
mt_4 = pd.DataFrame(matriz_confusao[3])
mt_5 = pd.DataFrame(matriz_confusao[4])
mt_6 = pd.DataFrame(matriz_confusao[5])
mt_7 = pd.DataFrame(matriz_confusao[6])


plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
