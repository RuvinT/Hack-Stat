# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:22:47 2019

@author: Ruvin Thulana
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



data = pd.read_csv("./hackstat2k19/Trainset.csv") 
datat = pd.read_csv("./hackstat2k19/xtest.csv")
# Preview the first 5 lines of the loaded data 
data = data.dropna()
X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16]].values

datat = datat.dropna()
Xt=datat.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,14,15,16,17]].values
print(X[0,12])
Y=data.iloc[:,17].values

labelencoder_X1 = LabelEncoder()
X[:, 10] = labelencoder_X1.fit_transform(X[:, 10])#month
Xt[:, 10] = labelencoder_X1.fit_transform(Xt[:, 10])
labelencoder_X4 = LabelEncoder()
print(X[13])
X[:, 13] = labelencoder_X4.fit_transform(X[:, 13])#ret vis
print(X[13])
Xt[:, 13] = labelencoder_X4.fit_transform(Xt[:, 13])
labelencoder_X2 = LabelEncoder()
X[:, 14] = labelencoder_X2.fit_transform(X[:, 14])#week
Xt[:, 14] = labelencoder_X2.fit_transform(Xt[:, 14])
labelencoder_X3 = LabelEncoder()
X[:, 11] = labelencoder_X3.fit_transform(X[:, 11])#province
Xt[:, 11] = labelencoder_X3.fit_transform(Xt[:, 11])
onehotencoder = OneHotEncoder(categorical_features = [10,13,11])  #13,11
X = onehotencoder.fit_transform(X).toarray()
Xnew=X
X=X[:,[1,2,3,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]




onehotencoder = OneHotEncoder(categorical_features = [10,13,11])
Xt = onehotencoder.fit_transform(Xt).toarray()
Xtnew=Xt
Xt=Xt[:,[1,2,3,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]




#######
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
mm_scaler = StandardScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test=mm_scaler.transform(X_test)
Xt=mm_scaler.transform(Xt)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

feature_cols = [ 'first', 'sec', 'thi','four','five','six','sev', 'eig', 'nine', 'ele','12','15','16','17','18','19','21','22','23','24','dg','rr','gh','ff','hf','rrryr''23','24','dg','rr','gh','ff']
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
            max_features=15, max_leaf_nodes=10, min_samples_leaf=9,
            min_samples_split=5, min_weight_fraction_leaf=0.1,
            presort=False, random_state=100, splitter='best')

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(Xt)
y_pred1 = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred1))




from IPython.display import Image  
import pandas as pd
import pydot
import sklearn.datasets as datasets

from sklearn.tree import export_graphviz


dot_data = export_graphviz(clf, out_file=None,
                filled=True, rounded=True,
                special_characters=True,
                feature_names = feature_cols,class_names=[ 'first', 'sec', 'thi','four','five','six','sev', 'eig', 'nine', 'ele','12','15','16','17','18','19','21','22','23'])
(graph,) = pydot.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('diagram.png')
y_pred=(y_pred>0.5)
arr=[]
for i in range(len(y_pred)):
    num=0
    if(y_pred[i]):
        num=1
    arr.append(num)
    


    
df = pd.DataFrame(arr) 

# saving the dataframe 
df.to_csv('sample_submisison.csv')


