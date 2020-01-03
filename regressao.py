from sklearn.linear_model import LinearRegression

#[anos de experiencia,  idade]
caracteristicas = [[10, 40], [2, 18], [4, 25], [5, 30], [4, 29], [6, 23], [8, 30], [7, 42]]
salarios = [15000, 3000, 4350, 2500, 8200, 5000, 7000, 7000]


regressor = LinearRegression()
            #Valor de Entrada          , Resultado
    
regressor.fit(caracteristicas, salarios)
salario = regressor.predict([[2, 40]])
print(salario)