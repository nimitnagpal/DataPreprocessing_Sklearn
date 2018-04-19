import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values  #Independent variables iloc[rows,colums] all three columns for all the 10 rows
Y = dataset.iloc[:, 3].values #Creating dependent variable which is in column 3

#Taking care of missing data - taking mean of the columns for which values not available using sci-kit library

from sklearn.preprocessing import Imputer #Importing Imputer class from sklearn library
imputer = Imputer(missing_values = 'NaN', strategy  = 'mean', axis = 0) #Creating object imputer linked with class Imputer
imputer = imputer.fit(X[:, 1:3]) # As the missing data is only in the columns 1 : 2 but due to upper bound for limit : becomes 3
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data - Giving unique entries to every category data for a column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()  #Onehotencoder so that, there is not confusion between dummy encoder values and their actual degree
labelencoder_y = LabelEncoder()  #Table Y encoded as 1 and 0 for Yes and No
Y = labelencoder_y.fit_transform(Y)

#Splitting dataset into training and test data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Scaling all the independent variables and also dummy encoders(3 columns)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # As there's no need to fit and then transform test data