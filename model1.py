import keras

from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout

import numpy as np
m1=m2=m3=m4=np.zeros((4,4))
data_train=np.load("256train.npy")
data_train = [i+1 for i in data_train]
data_train=np.reshape(data_train,(-1,17))
y_train=data_train[:,16]
#print(len(y_train))
x_train=data_train[:,0:16]
x_train=np.reshape(x_train,(-1,4,4))
x_train_new=np.zeros((len(x_train),8,8))
for i in range(len(x_train)):
    m1=x_train[i]
    m2=m1.T
    for j in range(4):
        m3[j]=x_train[i][3-j]
    m4=m3.T
    x_train_new[i]=np.concatenate((np.concatenate((m1, m2), axis=0), np.concatenate((m4, m3), axis=0)), axis=1)
x_train_new=np.reshape(x_train_new,(-1,8,8,1))

data_test=np.load("256.npy")
data_test= [i+1 for i in data_test]
data_test=np.reshape(data_test,(-1,17))
y_test=data_test[:,16]
x_test=data_test[:,0:16]
x_test=np.reshape(x_test,(-1,4,4))
x_test_new=np.zeros((len(x_test),8,8))
for i in range(len(x_test)):
    m1=x_train[i]
    m2=m1.T
    for j in range(4):
        m3[j]=x_test[i][3-j]
    m4=m3.T
    x_test_new[i]=np.concatenate((np.concatenate((m1, m2), axis=0), np.concatenate((m4, m3), axis=0)), axis=1)
x_test_new=np.reshape(x_test_new,(-1,8,8,1))

y_train=[i-1 for i in y_train]
y_train=keras.utils.to_categorical(y_train)
y_test=[i-1 for i in y_test]
y_test=keras.utils.to_categorical(y_test)

model=Sequential()
model.add(Conv2D(64,kernel_size=(4,1),strides=1,padding='valid',activation='relu',input_shape=(8,8,1)))
model.add(Conv2D(128,kernel_size=(1,4),strides=1,padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(2,2),strides=1,padding='same',activation='relu'))
model.add(Conv2D(200,kernel_size=(2,2),strides=1,padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(120,activation='softmax'))
model.add(Dense(64,activation='softmax'))
model.add(Dense(4,activation='softmax'))
model.summary()
model.compile('sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_new,y_train,batch_size=1,epochs=30,validation_data=[x_test_new,y_test])
model.save('model_256.h5')

