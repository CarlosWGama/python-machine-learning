from sklearn.cluster import KMeans

#caracteristicas de cada pessoa
pessoas = [
    [10, 2, 5],
    [10, 4, 8],
    [5, 8, 9],
    [2, 7, 9],
    [6, 3, 2],
    [1, 6, 3],
    [2, 9, 0],
    [7, 5, 2]
]

grupos = [[],[]]

#N_CLUSTERS = Número de grupos que deseja criar
#random_state = posição inicial aleatória
kmeans = KMeans(n_clusters=2, random_state=0)
resultado = kmeans.fit_predict(pessoas)

for i in range(len(resultado)):
    grupos[resultado[i]].append(pessoas[i])        

#Em que grupo está cada valor
print(grupos)

#Posição de cada centroide
print(kmeans.cluster_centers_)

grupo = kmeans.predict([[6, 2, 3]])
print(grupo)
