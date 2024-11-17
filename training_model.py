import tensorflow as tf
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np


dataDict = pickle.load(open('/Users/tahashahid/Rough-Neural-Network/dataset.pickle', 'rb'))
data = np.array(dataDict['data'], dtype=np.float32)
labels = np.array(dataDict['labels'], dtype=np.int32)


dataTrain, dataTest, labelsTrain, labelsTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
dataTrain = dataTrain.astype(np.float32)
labelsTrain = labelsTrain.astype(np.int32)


model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),     # Input layer
    Dropout(0.2),
    Dense(32, activation='relu'),                        # Hidden layer (where the processing happens)
    Dropout(0.2),
    Dense(3, activation='softmax')                       # Output layer

])


model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    dataTrain, 
    labelsTrain, 
    epochs=50, 
    batch_size=64, 
    validation_split=0.2,
    callbacks=[early_stopping]
)


loss, accuracy = model.evaluate(dataTest, labelsTest)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('RoughNeuralNetwork.h5')
