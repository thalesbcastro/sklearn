#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:23:19 2018

@author: thales
"""

import numpy as np 

def sigmoid(soma):
    # Função de ativação sigmoid
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    # Derivada da função sigmoide
    return sig * (1 - sig)    

    
entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],[1],[1],[0]]) 

#pesos0 = np.array([[-.424, -.740, -.961],
#                  [.358, -.577, -.469]])
#
#pesos1 = np.array([[-.017],[-.893],[.148]])

pesos0 = 2*np.random.random((2,3))-1 # nº de neurônios da camada de entrada e oculta: 2, 3
pesos1 = 2*np.random.random((3,1))-1 # nº neurônios camada oculta e saída: 3, 1

epocas = 10000
taxaAprendizagem = 0.3
momento = 1
for i in range(epocas):
    # Processo feed forward 
    camadaEntrada = entradas
    somaSinapse0 = np.dot(entradas, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    # Cálculo do erro de maneira simples: valor da classe - o valor calculado
    # Pega-se a média desse valor
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    #Cálculo do Delta da camada de saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    #Cálculo do Delta da camada escondida deltaSaida = derivadaOculta * pesosSaida * deltaSaida
    pesos1Transposta = pesos1.T 
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta) # pesos * deltaSaida 
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    # ALGORITMO BACKPROPAGATION 
    # Mudança dos pesos da camada oculta para a camada de saída 
    # peso(n+1) = (peso * momento) + (entrada * delta * taxa de aprendizagem) 
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem) 
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
    
    '''
        O número de classes que eu tenho, será o número de neurônios na camada de saída.
        Por exemplo: 
            ativo      - 0,0,0,1 (valores esperados para cada neurônio);
            reflexivo  - 0,0,1,0; 
            pragmático - 0,0,1,1;
            e outro.   - 0,1,0,0.
        
        O número da camada oculta é geralmente camada_oculta = (entrada + saidas)/2.
        
        A camada de entrada é determinada pelo nº de atributos que vou passar para o treinamento
    '''