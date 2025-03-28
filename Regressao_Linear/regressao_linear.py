from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Cria um dataset chamado 'df' que carregará os dados do csv
df = pd.read_csv("C:/Users/jhona/OneDrive/Projetos/IA/Regressao_Linear/FuelConsumptionCo2.csv")

#EXIBE A ESTRUTURA DO DATAFRAME
print(df.head())

"""# Exibe o resumo do Dataset"""

print(df.describe())

"""# Selecionar apenas as features do Motor e CO2"""

motores =  df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())

"""#Dividir o dataset em dados de treinamento e dados de teste
neste casos vamos usar o train_test_split do scikitlearn
"""

motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)
print(type(motores_treino))

"""#Exibir a correlação entre as features do dataset de treinamento"""

plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()

"""# Vamos treinar o modelo de regressão linear"""

# CRIAR UM MODELO DE TIPO DE REGRESSÃO LINEAR
modelo =  linear_model.LinearRegression()

# TREINAR O MODELO USANDO O DATASET DE TESTE
# PARA ENCONTRAR O VALOR DE A E B (Y = A + B.X)
modelo.fit(motores_treino, co2_treino)

"""#Exibir os coeficientes (A e B)"""

print('(A) Intercepto: ', modelo.intercept_)
print('(B) Inclinação: ', modelo.coef_)

"""# Vamos exibir a nossa reta de regressão no dataset de treino"""

plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0]*motores_treino + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

"""# Vamos executar o nosso modelo no dataset de teste"""

#Primeiro a gente tem que fazer as predições usando o modelo e base de teste
predicoesCo2 = modelo.predict(motores_test)

"""# Vamos exibir a nossa reta de regressão no dataset de teste"""

plt.scatter(motores_test, co2_teste, color='blue')
plt.plot(motores_test, modelo.coef_[0][0]*motores_test + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

"""# Vamos avaliar o modelo"""

#Agora é mostrar as métricas
print("Soma dos Erros ao Quadrado (SSE): %.2f " % np.sum((predicoesCo2 - co2_teste)**2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_teste, predicoesCo2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_teste, predicoesCo2))
print ("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print("R2-score: %.2f" % r2_score(co2_teste, predicoesCo2) )