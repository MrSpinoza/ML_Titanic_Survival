# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
print(df.head())

# Cleaning Data. Sex feature to numerical; Age nan to average; First/Second/Third Class
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age']= df['Age'].fillna(df['Age'].mean())
df['FirstClass'] = df.Pclass.apply(lambda x: 1 if x == 1 else 0)
df['SecondClass'] = df.Pclass.apply(lambda x: 1 if x == 2 else 0)
df['ThirdClass'] = df.Pclass.apply(lambda x: 1 if x == 3 else 0)

# Select and Split the Data
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survived = df[['Survived']]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, survived, train_size=0.2, test_size=0.8 )

# Normalize the Data. 
# Since sklearn‘s Logistic Regression implementation uses Regularization, we need to scale our feature data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and Evaluate the Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
# The score returned is the percentage of correct classifications, or the accuracy.
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

# The feature coefficients determined by the model.
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))


# Predict with the Model
# Sample passenger features

Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Bojan = np.array([0.0,37,0.0,1.0])
Jelena = np.array([1.0,38,0.0,1.0])
Jana = np.array([1.0,9,0.0,1.0])
Lena = np.array([1.0,1,0.0,1.0])
sample_passangers = np.array([Jack, Rose, Bojan, Jelena, Jana, Lena])
print(sample_passangers)
# Since our Logistic Regression model was trained on scaled feature data, we must also scale the feature data we are making predictions on.
sample_passangers= scaler.transform(sample_passangers)
print(sample_passangers)

print(model.predict(sample_passangers))
print(model.predict_proba(sample_passangers))
