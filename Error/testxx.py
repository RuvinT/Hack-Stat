# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:28:33 2019

@author: Ruvin Thulana
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:22:47 2019

@author: Ruvin Thulana
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



data = pd.read_csv("./Trainset.csv") 
datat = pd.read_csv("./xtest.csv")
# Preview the first 5 lines of the loaded data 
data = data.dropna()
X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14,15,16]].values

Y=data.iloc[:,16].values

labelencoder_X1 = LabelEncoder()
X[:, 10] = labelencoder_X1.fit_transform(X[:, 10])

labelencoder_X2 = LabelEncoder()
X[:, 12] = labelencoder_X2.fit_transform(X[:, 12])
labelencoder_X3 = LabelEncoder()
X[:, 13] = labelencoder_X3.fit_transform(X[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [10,12])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

datat = datat.dropna()
Xt=datat.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,15,16,17]].values



labelencoder_X1t = LabelEncoder()
Xt[:, 10] = labelencoder_X1t.fit_transform(Xt[:, 10])

labelencoder_X2t = LabelEncoder()
Xt[:, 12] = labelencoder_X2t.fit_transform(Xt[:, 12])
labelencoder_X3t = LabelEncoder()
Xt[:, 13] = labelencoder_X3t.fit_transform(Xt[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [10,12])
Xt = onehotencoder.fit_transform(Xt).toarray()
Xt=Xt[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]



#######
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,shuffle=False)

from sklearn.preprocessing import StandardScaler
mm_scaler = StandardScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test=mm_scaler.transform(X_test)
Xt=mm_scaler.transform(Xt)
print(len(X_train[0])) 
print(len(Xt[0]))  
for i in range(len(X_train)):
    for j in range(len(X_train[0])):
        if(X_train[i][1]==Xt[i][1]):
            print("true")
        else:
            print("false")
            break
        
#results = confusion_matrix(Y_test, y_pred)
#print ('Accuracy Score :',accuracy_score(Y_test, y_pred) )







