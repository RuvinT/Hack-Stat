# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:22:47 2019

@author: Ruvin Thulana
"""
#import plaidml.keras
#plaidml.keras.install_backend()


import keras

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



data = pd.read_csv("./hackstat2k19/Trainset.csv") 
datat = pd.read_csv("./hackstat2k19/xtest.csv")
# Preview the first 5 lines of the loaded data 
data = data.dropna()
X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,14,15,16]].values

Y=data.iloc[:,17].values

labelencoder_X1 = LabelEncoder()
X[:, 10] = labelencoder_X1.fit_transform(X[:, 10])

labelencoder_X2 = LabelEncoder()
X[:, 12] = labelencoder_X2.fit_transform(X[:, 12])
labelencoder_X3 = LabelEncoder()
X[:, 13] = labelencoder_X3.fit_transform(X[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [10,12])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23]]

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
Xt=Xt[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23]]



#######
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


from sklearn.preprocessing import StandardScaler
mm_scaler = StandardScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test=mm_scaler.transform(X_test)
Xt=mm_scaler.transform(Xt)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

optimizers = [ 'adam']
init = [  'uniform']
epochs = np.array([50, 100, 150,300,400,500])
batches = np.array([4,5, 10,12, 20,25,32,45])
#optimizers = ['adam']
#init = ['normal']
#epochs = np.array([ 10])
#batches = np.array([5])

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(output_dim=30,init=init, activation='relu', input_dim =22))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=6,init=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1,init=init, activation='sigmoid'))
    model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy']) 
    return model



model = KerasClassifier(build_fn=create_model)
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)




modeln =Sequential()

modeln.add(Dense(output_dim=6,init=grid_result.best_params_['init'], activation='relu', input_dim =22))
modeln.add(Dense(output_dim=6,init=grid_result.best_params_['init'], activation='relu'))

modeln.add(Dense(output_dim=1,init=grid_result.best_params_['init'], activation='sigmoid'))
modeln.compile(optimizer=grid_result.best_params_['optimizer'], 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


modeln.fit(X_train, Y_train,batch_size=grid_result.best_params_['batch_size'], nb_epoch = grid_result.best_params_['nb_epoch'])


y_pred=modeln.predict(Xt)

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

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix   

#results = confusion_matrix(Y_test, y_pred)
#print ('Accuracy Score :',accuracy_score(Y_test, y_pred) )







