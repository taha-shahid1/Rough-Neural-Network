import mediapipe as mp
import pickle
import os
import cv2 as cv

DATA_DIR = '/Users/tahashahid/Rough-Neural-Network/dataset'

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for imgPath in os.listdir(os.path.join(DATA_DIR, dir)):
        dataBuffer = []
        xCoord = []
        yCoord = []

        img = cv.imread(os.path.join(DATA_DIR, dir, imgPath))
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results =  hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                    
                    xCoord.append(x)
                    yCoord.append(y)
                    for i in range(len(landmarks.landmark)):
                        x = landmarks.landmark[i].x
                        y = landmarks.landmark[i].y
                        dataBuffer.append(x - min(xCoord))
                        dataBuffer.append(y - min(yCoord))
            
            data.append(dataBuffer)
            labels.append(dir)
        
picklerickfile = open('dataset.pickle', 'wb')
pickle.dump({'labels': labels, 'data': data}, picklerickfile)
picklerickfile.close()