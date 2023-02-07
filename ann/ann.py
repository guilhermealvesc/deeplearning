import numpy as np
import pandas as pd
import tensorflow as tf

# print(tf.__version__) # 2.11.0

# Importando os dados

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values  # 3 coluna até a penúltima
y = dataset.iloc[:, -1].values  # última coluna

# print(x)
# print(y)

# Codificando os dados categóricos
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Transformada os dados da coluna de gênero em dado categorico
# indexação [linha, coluna] e [:, 2] significa todas as linhas e a coluna 2
x[:, 2] = le.fit_transform(x[:, 2])

print(x)
