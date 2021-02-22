import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r'04_dados_exercicio.csv')

features = dataset.iloc[:, :-1].values

classe = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(features[:, 2:4])
features[:, 2:4] = imputer.transform(features[:, 2:4])
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features))
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)

features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(features, classe, test_size = 0.15, random_state=1)

standardScaler = StandardScaler()

features_treinamento[:, 4:6] = standardScaler.fit_transform(features_treinamento[:, 4:6])

features_teste[:, 4:6] = standardScaler.transform(features_teste[:, 4:6])

print('====================features=====================')
print(features)
print('==============features_treinamento===============')
print(features_treinamento)
print('=================features_teste==================')
print(features_teste)