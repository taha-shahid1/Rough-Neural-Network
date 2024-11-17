import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model('/Users/tahashahid/Rough-Neural-Network/RoughNeuralNetwork.h5')


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)


cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            dataBuffer = []
            xCoord = []
            yCoord = []

            # Collect all coordinates for normalization
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                xCoord.append(x)
                yCoord.append(y)

            # Normalize and store coordinates
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                dataBuffer.append(x - min(xCoord))
                dataBuffer.append(y - min(yCoord))

            
            dataBuffer = np.array(dataBuffer).reshape(1, -1)
            predictions = model.predict(dataBuffer, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            
            cv2.putText(frame, 
                       f"Class: {predicted_class} Conf: {confidence:.2f}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
            
            
            for i, landmark in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()