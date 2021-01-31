from keras.datasets import imdb
from keras import models
from keras.layers import Dense
import numpy as np  
import matplotlib.pyplot as plt

(XT,YT),(Xt,Yt)=imdb.load_data(num_words=10000)
indx_word = dict([(val, key) for key,val in word_indx.items()])

def vectorize_sentences(sentences,dims=10000):
  outputs = np.zeros((len(sentences),dims))
  for i,indx in enumerate(sentences):
    outputs[i,indx] = 1 
  return outputs

X_train = vectorize_sentences(XT)
X_test = vectorize_sentences(Xt)  
Y_train = np.asarray(YT).astype('float32')
Y_test = np.asarray(Yt).astype('float32')

model = models.Sequential()
model.add(Dense(16,activation='relu',input_shape=(10000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

X_val = X_train[:5000]
X_train_new = X_train[5000:]
Y_val = Y_train[:5000]
Y_train_new = Y_train[5000:]

model.fit(X_train_new,Y_train_new,epochs=4,batch_size = 512,validation_data=(X_val,Y_val))