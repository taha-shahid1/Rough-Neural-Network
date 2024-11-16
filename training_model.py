import tensorflow as tf
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np


dataDict = pickle.load(open('/Users/tahashahid/Rough-Neural-Network/dataset.pickle', 'rb'))
data = np.array(dataDict['data'], dtype=np.float32)
labels = np.array(dataDict['labels'], dtype=np.int32)


dataTrain, dataTest, labelsTrain, labelsTest = train_test_split(data, labels, test_size=0.2, shuffle = True, stratify = labels)
dataTrain = dataTrain.astype(np.float32)
labelsTrain = labelsTrain.astype(np.int32)


model = Sequential([
    Dense(64, activation='relu', input_shape=(data.shape[1],)),  # Input layer
    Dropout(0.2),  # Regularization
    Dense(32, activation='relu'),  # Hidden layer (where the processing happens)
    Dropout(0.2),
    Dense(3, activation='softmax')  # Output layer for 3 alphabets (I did 3 only because im too lazy to capture images for every letter, this was just to test if it works for the first 3 letters)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # On idrees' soul we're finished
              metrics=['accuracy'])

# Model training
model.fit(dataTrain, labelsTrain, epochs=50, batch_size=64, validation_split=0.2)

# Testing
loss, accuracy = model.evaluate(dataTest, labelsTest)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100}")


model.save('RoughNeuralNetwork.h5')