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
    Dense(64, activation='relu', input_shape=(42,)),     # Input layer, which uses relu. If the output of the function is positive, the function's output will remain unchanged. However, if it is negative, relu makes it 0
    Dropout(0.2),                                        # During training, dropout randomly sets a fraction of the neurons (0.20 of neurons here) in a layer to zero at each training step which forces the model to rely on different combinations of neurons and reduces its reliance on specific neurons
    Dense(32, activation='relu'),                        # Hidden layer (where the processing happens)
    Dropout(0.2),
    Dense(3, activation='softmax')                       # Output layer, softmax means the outputs will be in the form of probabilities of which label the landmarks should belong to

])


model.summary()


model.compile(optimizer='adam',                                # Adaptive Moment Estimation adjusts weights efficiently by remembering past updates and adapting step sizes (how much weights are changed in each iteration) for each weight, helping the model learn faster and more steadily
              loss='sparse_categorical_crossentropy',          # Sparse Categorical Crossentropy is a loss function for classification when the labels are integers. It compares predicted probabilities with the true class and penalizes low probabilities for the correct class
              metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(             # Prevents over-fitting by stopping training when the models performance on the validation/test data stops improving
    monitor='val_loss',                                        # Tracks validation loss
    patience=10,                                               # Stops training if the validation loss doesn’t improve for 10 consecutive epochs
    restore_best_weights=True                                  # Ensures the model returns to the state (with correct weights assigned to each neuron) with the best validation loss before stopping
)

model.fit(
    dataTrain, 
    labelsTrain, 
    epochs=50,                                                 # An epoch is one complete pass through the entire training dataset by the model. During each epoch, the model processes all the training samples, computes the loss for each, and updates its parameters using the optimizer
    batch_size=64,                                             # Number of training samples processed before updating each neurons weights in one iteration
    validation_split=0.2,
    callbacks=[early_stopping]
)


loss, accuracy = model.evaluate(dataTest, labelsTest)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('RoughNeuralNetwork.h5')
