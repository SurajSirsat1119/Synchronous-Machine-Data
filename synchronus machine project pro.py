##Data Set Characteristics= Multivariate
##y = f(x)
##y = target colum (557)(Number o output)
##x = input colums(557,4)(Number of Instances)
##input classes = [Iy,PF,e,dIf](Number of Attributes)
##Associated Tasks=Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('synchronous machine.csv',delimiter=';',decimal=',')
##print(df.head(30))
print(np.shape(df))
x  = df.iloc[:,:-1]
print(np.shape(x))
##print(x.head(30))

y = df.iloc[:,-1]
##print(y.head())
print(np.shape(y))

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.45)
##print(X_train)
print(np.shape(X_test))
print(X_test[:10])
##print(y_train)
##print(y_test)

scale = MinMaxScaler()
X_train_new = scale.fit_transform(X_train)
##print(X_train_new)
X_test_new = scale.transform(X_test)
##print(X_test_new)

#traning model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(X_train_new,y_train)
score = model.score(X_train_new,y_train)
print('score =',score)


##print(model.predict(X_test_new[:10]))
print(y_test[:10])

#testing
test = [3.7 , 0.98,  0.02,  0.140  ]
test =np.matrix(test)
test = scale.transform(test)
result = model.predict(test)
print(result)
print(float('%.3f' %result))

pipe = Pipeline(steps =[
    ('step1',scale),
    ('step2',LinearRegression())
    ])
pipe.fit(X_train_new,y_train)
y_pred = pipe.predict(X_test_new)
##print(y_pred)
##print(pipe.score(X_train_new,y_train))
import pickle
####pickle.dump(scale,open('scale.pkl','wb'))
####pickle.dump(model,open('model.pkl','wb'))
##pickle.dump(pipe,open('pipe.pkl','wb'))
##pickle.dump(y_train,open('y_train.pkl','wb'))
##pickle.dump(X_train_new,open('X_train.pkl','wb'))
