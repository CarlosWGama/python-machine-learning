#https://www.vooo.pro/insights/6-passos-faceis-para-aprender-o-algoritmo-naive-bayes-com-o-codigo-em-python/
from enum import Enum;

#CARACTERISTICAS
TEMPO_CHOVEU = 1
TEMPO_SOL = 2
TEMPO_NUBLADO = 3

DINHEIRO_SIM = 1
DINHEIRO_NAO = 2

#CLASSIFICADOR
PRAIA_FOI = 1
PRAIA_NAO_FOI = 0

caracteristicas = [
    [TEMPO_CHOVEU, DINHEIRO_NAO],
    [TEMPO_CHOVEU, DINHEIRO_SIM],
    [TEMPO_CHOVEU, DINHEIRO_NAO],
    [TEMPO_NUBLADO, DINHEIRO_NAO],
    [TEMPO_NUBLADO, DINHEIRO_NAO],
    [TEMPO_NUBLADO, DINHEIRO_SIM],
    [TEMPO_NUBLADO, DINHEIRO_NAO],
    [TEMPO_SOL, DINHEIRO_SIM],
    [TEMPO_SOL, DINHEIRO_SIM],
    [TEMPO_SOL, DINHEIRO_NAO],
    [TEMPO_SOL, DINHEIRO_NAO]
]

resultado = [
    PRAIA_NAO_FOI, #Não foi
    PRAIA_NAO_FOI, #Não foi
    PRAIA_NAO_FOI, #Não foi
    PRAIA_NAO_FOI, #Não foi
    PRAIA_NAO_FOI, #Não foi
    PRAIA_FOI, #Foi a praia
    PRAIA_FOI, #Foi a praia
    PRAIA_FOI, #Foi a praia
    PRAIA_FOI, #Foi a praia
    PRAIA_FOI, #Foi a praia
    PRAIA_NAO_FOI #Não foi
]

NaiveBayes = False
DecisionTree = True

if (NaiveBayes): 
    from sklearn.naive_bayes import GaussianNB

    modelo = GaussianNB()

    #Treinando o modelo
    modelo.fit(caracteristicas, resultado)

    #Prevendo
    previsao = modelo.predict([
        [TEMPO_SOL, DINHEIRO_SIM],
        [TEMPO_SOL, DINHEIRO_SIM],
        [TEMPO_NUBLADO, DINHEIRO_SIM]
    ])

    print(previsao)
elif (DecisionTree):
    #https://pypi.org/project/dtreeplt/
    from sklearn import tree
    from dtreeplt import dtreeplt

    #pip install matplot

    modelo = tree.DecisionTreeClassifier()
    modelo = modelo.fit(caracteristicas, resultado)

    #Prevendo
    previsao = modelo.predict([
        [TEMPO_SOL, DINHEIRO_SIM],
        [TEMPO_SOL, DINHEIRO_SIM],
        [TEMPO_CHOVEU, DINHEIRO_SIM]
    ])
    
    print(previsao)

    #Arvore 
    nome_campos=['Tempo', 'Dinheiro']
    nome_resultados = ['Não vai', 'Vai a praia']
    
    dtree = dtreeplt(model=modelo, feature_names=nome_campos, target_names=nome_resultados)
    fig = dtree.view()
    #if you want save figure, use savefig method in returned figure object.
    fig.savefig('output.png')

    #Sample - Quantos valores existem a serem identificados
    #Values - QUantos valores tem em cada classificador na amostra
    #Gini - Porcentagem para um dos valores
    #Transparente - Não é certeza absoluta
    #Opaco - CErteza absoluta


