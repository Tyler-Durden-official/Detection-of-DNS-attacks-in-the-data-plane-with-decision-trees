from keras.models import Sequential
from keras.layers import *
from keras.layers.embeddings import Embedding

#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt
#import seaborn as sn
from sklearn.metrics import confusion_matrix

data=pd.read_csv("tokenised1.csv")

x_total=data.iloc[:,:-1]
y_total=data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x_total,y_total,test_size=0.3)

#def multiclass_model(input_shape=(200,20), filters, size_kernel,dense1_size, dense2_size, dense3_size):
model = Sequential()
model.add(Embedding(input_dim=200, output_dim=100, input_length=20))

model.add(Convolution1D(filters=512,kernel_size=4,strides=1,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
optimizer='Adam',metrics=['accuracy'])
  


#model=multiclass_model(512,4,256,1024,256,0.5)
model.fit(x_train,y_train,epochs=10)
print(model.evaluate(x_test,y_test))