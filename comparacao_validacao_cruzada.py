# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
# load dataset
from sklearn.metrics import accuracy_score, confusion_matrix

'''
    LabelEncoder transforma os labels de texto das classes em númericos. E para voltar aos labels iniciais
    é só LabelEncoder.inverse_transform(y)
'''
from sklearn.preprocessing import LabelEncoder

'''
    https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/
'''
dados = pd.read_csv('./testes_outros_algoritmos.csv')
'''
    Descobri que todo o código abaixo pode ser substituído usando LabelEncoder
'''
'''
dados.loc[dados['estilo_de_aprendizagem']=='Indefinido','estilo_de_aprendizagem'] = 0
dados.loc[dados['estilo_de_aprendizagem']=='Ativo',     'estilo_de_aprendizagem'] = 1
dados.loc[dados['estilo_de_aprendizagem']=='Teorico',   'estilo_de_aprendizagem'] = 2
dados.loc[dados['estilo_de_aprendizagem']=='Reflexivo', 'estilo_de_aprendizagem'] = 3
dados.loc[dados['estilo_de_aprendizagem']=='Pragmatico','estilo_de_aprendizagem'] = 4
'''
array = dados.values
X = array[:, :14]
y = array[:, 14:15]
label = LabelEncoder()
y = label.fit_transform(y)
#y = label.inverse_transform(y)
y = y.ravel()
# prepare configuration for cross validation test harness
seed = 7
# prepare models
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier

lgr_ovr = LogisticRegression(multi_class='ovr')
lgrCV_ovr = LogisticRegressionCV(multi_class='ovr', random_state=None)
#mlp = MLPClassifier(verbose=0, random_state=0, hidden_layer_sizes=(100, 250),
#                    max_iter=200, solver= 'adam', learning_rate_init= 0.01)
#mlp = MLPClassifier(verbose=0, solver= 'sgd', learning_rate= 'adaptive', 
#                    hidden_layer_sizes=(150, 260),
#                    learning_rate_init= 0.1, momentum= 0.8, nesterovs_momentum= True, 
#                    activation= 'tanh', early_stopping=True, max_iter=400, random_state=0)
models = []

'''
    IMPORTANTE!! IMPORTANTE!! IMPORTANTE!! OLHAR!! OLHAR!!!
    Melhores parâmetros para minha MLP para o meu conjunto de dados:
    https://stackoverflow.com/questions/46028914/multilayer-perceptron-convergencewarning-stochastic-optimizer-maximum-iterat
'''
'''
    PROCURAR SABER MAIS DOS MÉTODOS: OVO, OVR e Output-Code:
    https://scikit-learn.org/stable/modules/multiclass.html#ovr-classification. Do It.
'''
'''
    Verificou-se que para a MLP não há muita diferença entre usar o StratifiedKFold ou o KFold. Além
    disso, observou-se que os melhores resultados foram obtidas com o OVR.
'''
models.append(('1-MLP DEFAULT', MLPClassifier(random_state=0)))
models.append(('1-MLP OVO', OneVsOneClassifier(MLPClassifier(random_state=0))))
models.append(('1-MLP OVR', OneVsRestClassifier(MLPClassifier(random_state=0))))
models.append(('1-MLP OutputCodeClassifier', OutputCodeClassifier(MLPClassifier(random_state=0))))
#
'''
    O LogisticRegression apresentou melhores resultados com o StratifiedKFold, observando que setando multi_class para
    'multinomial' apresenta melhores resultados que os outros: OVO, OVR e OutputCodeClassifier. Além disso, observa-se que para este 
    último, apresenta-se o pior resultado.
'''
#models.append(('2-LgReg DEFAULT', LogisticRegression(solver='lbfgs', multi_class='multinomial')))
#models.append(('2-LgReg OVO', OneVsOneClassifier(LogisticRegression(solver='lbfgs'))))
#models.append(('2-LgReg OVR', OneVsRestClassifier(LogisticRegression(solver='lbfgs'))))
#models.append(('2-LgReg OutputCodeClassifier', OutputCodeClassifier(LogisticRegression(solver='lbfgs'))))
#
'''
    Assim como os anteriores, o  LogisticRegressionCV apresentou melhores resultados com 
    o StratifiedKFold, no entanto, diferentemente do anterior, este apresentou melhores resultados
    com o OVO e OVR.
'''
#models.append(('3-LgRegCV DEFAULT', LogisticRegressionCV(multi_class='multinomial')))
#models.append(('3-LgRegCV OVO', OneVsOneClassifier(LogisticRegressionCV())))
#models.append(('3-LgRegCV OVR', OneVsRestClassifier(LogisticRegressionCV())))
#models.append(('3-LgRegCV OutputCodeClassifier', OutputCodeClassifier(LogisticRegressionCV())))

'''
    o SVC utiliza por padrão a estratégia OVO e apresentou melhores resultados com o StrifierKfold. Percebe-se que
    o OVO é ligeiramente superior ao OVR, sendo o Output-code o menor entre eles.
'''
#models.append(('4-SVM DEFAULT', SVC()))
#models.append(('4-SVM OVO', OneVsOneClassifier(SVC())))
#models.append(('4-SVM OVR', OneVsRestClassifier(SVC())))
#models.append(('4-SVM OutputCodeClassifier', OutputCodeClassifier(SVC())))
'''
    Por fim, para o KNeighborsClassifier foi verificado que não houve uma diferença significativa
    nas métricas, utilizando ou o StratifiedKFold ou o KFold. Além disso, o melhor resultado se encontra
    quando se utiliza a estratégia OVO.
'''
#models.append(('(5)KNN DEFAUL', KNeighborsClassifier()))
#models.append(('(5)KNN OVO', OneVsOneClassifier(KNeighborsClassifier())))
#models.append(('(5)KNN OVR', OneVsRestClassifier(KNeighborsClassifier())))
#models.append(('(5)KNN', OutputCodeClassifier(KNeighborsClassifier())))


'''
    Modelos inicialmente testados no cru. 
'''
#models.append(('(0)MLP', MLPClassifier(random_state=0)))
##
#models.append(('(1)LR-multinomial', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')))
##models.append(('2 - LR-OVR', lgr_ovr))
##
#models.append(('(2)LRCV-multinomial', LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial')))
##models.append(('4 - lgrCV OVR', lgrCV_ovr))
##
##models.append(('5 - SVM default', OneVsRestClassifier(SVC())))
#models.append(('(3)SVM', SVC()))
#'''
#    Para que o LinearSVC possa classificar problemas Multiclasses, deve-se setar para "crammer_singer"
#'''
#models.append(('(4)LinearSVC-crammer_singer', LinearSVC(multi_class="crammer_singer")))
##models.append(('8 - LinearSVC-ovr', LinearSVC(multi_class="ovr")))
##
#models.append(('(5)KNN', KNeighborsClassifier()))
#models.append(('(6)TREE', DecisionTreeClassifier()))
###models.append(('9 - GPC-OVR', GaussianProcessClassifier(kernel=1.0 * RBF(1.0), multi_class='one_vs_rest')))
#models.append(('(7)GPC', GaussianProcessClassifier(kernel=1.0 * RBF(1.0), multi_class='one_vs_one')))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

def funcao_stratifierKfold(X, y):
    skf = StratifiedKFold(n_splits=10, random_state=seed)
    for train_index, test_index in skf.split(X, y):
        X_1, X_2 = X[train_index], X[test_index]
        y_1, y_2 = y[train_index], y[test_index]
    return X_1, X_2, y_1, y_2

def funcao_Kfold(X):
    kf = KFold(n_splits=10, random_state=seed)
    for train_index, test_index in kf.split(X):
        X_1, X_2 = X[train_index], X[test_index]
        y_1, y_2 = y[train_index], y[test_index]
    return X_1, X_2, y_1, y_2

def validacao_cruzada(models, X, y, n):
    for name, model in models:
        if n == 1:
            kfold = KFold(n_splits=10, random_state=7)
        else:
            kfold = StratifiedKFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    	  
        results.append(cv_results)
    	    
        names.append(name)
        
        msg = "%s:\n %f (%f)" % (name, cv_results.mean(), cv_results.std())
        
#        model.fit()
    	    
        print(msg)
    
def sem_validacao_cruzada(models, X, y):
    tec = int(input("(1) StratifierKfold \n(2) Kfold \n(3) Outro\n"))
    if tec == 1:
        X_train, X_test, y_train, y_test = funcao_stratifierKfold(X, y)
    elif tec ==2 :
        X_train, X_test, y_train, y_test = funcao_Kfold(X)
    else: 
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score_treino = model.score(X_train, y_train)
        accuracy_previsao = accuracy_score(y_test, y_pred)
#        matriz_confusao = confusion_matrix(y_test, y_pred) ou (y_pred, y_tes)??
    	
        msg1 = "Para %s, tem-se:\nTreino: %f " % (name, score_treino)
        msg2 = "Previsão: %f " %(accuracy_previsao)
#        msg3 = "Matriz de confusão:\n"
        
        print(msg1)
        print(msg2)
#        print(msg3)
#        print(matriz_confusao)


val = int(input("Usar validação cruzada: 1-sim / 0-não\n"))
if val == 1:
    n = int(input("(1) Kfold ou (2) StratifierKfold\n"))
    validacao_cruzada(models, X, y, n)
else:
    sem_validacao_cruzada(models, X, y)


# boxplot algorithm comparison
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()
    
corr = dados.corr()
corr.iloc[:, 14:15].sort_values(ascending=False)
dados.drop("estilo_de_aprendizagem", axis=1).apply(lambda x: x.corr(dados.Target))

