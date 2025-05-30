import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Config
gesture_name = "A"  # change for each gesture
save_dir = f"data/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
count = 0  # counter for saved images

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

        if imgCrop.size == 0:
            continue

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

        # Show and save
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        key = cv2.waitKey(1)
        if key == ord("s"):
            count += 1
            save_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(save_path, imgWhite)
            print(f"Saved {save_path}")
        elif key == ord("q"):
            break
    else:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
