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


from sklearn.preprocessing import StandardScaler
mm_scaler = StandardScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test=mm_scaler.transform(X_test)
Xt=mm_scaler.transform(Xt)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model =Sequential()

model.add(Dense(output_dim=20,init='uniform', activation='relu', input_dim =31))


model.add(Dense(output_dim=6,init='uniform', activation='relu'))

model.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


model.fit(X_train, Y_train,batch_size=5, nb_epoch =100)
y_pred1=model.predict(X_test)

y_pred1=(y_pred1>0.5)
y_pred=model.predict(Xt)

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix   

results = confusion_matrix(Y_test, y_pred1)
print ('Accuracy Score :',accuracy_score(Y_test, y_pred1) )
