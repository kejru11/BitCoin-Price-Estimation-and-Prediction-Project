#importing the tools for Exploratory Data Analysiis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the Data Manipulation Tools
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

#importing the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import ElasticNet

#extracting the data from the CSV file
data= pd.read_csv("BTC_Data_final.csv")
data["Date"]= data["Date"].str.replace("/", "")
data["Date"]= data["Date"].astype(int)
print(data.dtypes)

#Separating the Price column from the Dataset for Price Prediction Analysis
X= data.drop("priceUSD", axis=1)
y= data["priceUSD"]

#checking for missing values in the dataset
print(data.isna().sum())

#splitting the dataset in the training and testing datasets

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2)

#importing and training the modules on the dataset
#we have to predict a quantity, so I am training it on the Regression models

#importing the RidgeRegression Model
np.random.seed(42)
model= Ridge()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(cross_val_score(model, X, y, cv=5).mean())

#the above model without tuning any hyperparameters returns an accuracy of about 99.9%
#let's try to tune the model and find ways to improve the accuracy