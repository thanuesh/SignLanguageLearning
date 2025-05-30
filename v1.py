import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize video capture and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the updated Keras model
model = load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f]

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

        if imgCrop.size == 0:
            continue  # Skip if the crop is empty

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Preprocess the image for the model
        imgWhite = imgWhite.astype('float32') / 255.0

        # Resize the image to 150x150 as expected by the model
        # Resize the image to 224x224 as expected by the model
        imgWhite = cv2.resize(imgWhite, (224, 224))

        # Add batch dimension (for a single image, this is necessary)
        imgWhite = np.expand_dims(imgWhite, axis=0)

        # Make predictions
        prediction = model.predict(imgWhite)
        index = np.argmax(prediction)
        print(prediction, index)

        # Display results
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite[0])  # Remove batch dimension for display

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
