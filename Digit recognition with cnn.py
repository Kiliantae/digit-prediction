import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#reading the data in
test = pd.read_csv('/home/kilian/Desktop/Projects/test.csv')
train = pd.read_csv('/home/kilian/Desktop/Projects/train.csv')
print(test.shape)

#splitting training set into labels and images
train_x, test_x = train.iloc[:,1:].values , test.iloc[:,:].values
train_y = train['label']
print("the shape of train_x is: " , train_x.shape, "the shape of test_x is: ", test_x.shape
      , "\nthe shape of train_y is: ", train_y.shape,"\n")
#preprocess
train_X, test_X = train_x.reshape(42000,28,28,1) , test_x.reshape(28000,28,28,1)
train_Y  = to_categorical(train_y) #'one-hot-encode' the target
print("the shape of train_X is: " , train_X.shape, "the shape of test_X is: "
      , "\nthe shape of test_Y is:", train_Y.shape)

#split the data into train and validation set
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y , test_size=0.20, random_state=42)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)