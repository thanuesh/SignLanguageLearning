import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Constants
offset = 20
imgSize = 300
detector = HandDetector(maxHands=1)

# Streamlit App
st.set_page_config(page_title="ASL Learning App", layout="centered")
st.title("🧠 ASL Learning App (A - E)")

page = st.sidebar.selectbox("Go to", ["Learn", "Train", "Test"])

# ------ Learn Page ------
if page == "Learn":
    st.header("📘 Learn ASL (A - E)")
    letter = st.selectbox("Choose a letter:", ["A", "B", "C", "D", "E"])
    img_path = f"images/{letter}.png"
    st.image(img_path, width=300)

# ------ Train Page ------
elif page == "Train":
    st.header("💪 Train (Live Feedback)")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        success, img = cap.read()
        if not success:
            st.warning("Camera not accessible.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        result = "Show a hand gesture..."

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

            if imgCrop.size > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                imgWhite = imgWhite.astype('float32') / 255.0
                imgWhite = cv2.resize(imgWhite, (224, 224))
                imgWhite = np.expand_dims(imgWhite, axis=0)
                prediction = model.predict(imgWhite)
                index = np.argmax(prediction)
                result = f"Prediction: {labels[index]}"

        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(imgRGB)
        st.markdown(f"**{result}**")

# ------ Test Page ------
elif page == "Test":
    st.header("🧪 Test Yourself")

    test_words = [("B_D", "E"), ("_AIT", "B"), ("_ook", "C"), ("_ATE", "D")]
    current_question = st.session_state.get("current_question", 0)
    score = st.session_state.get("score", 0)

    st.subheader(f"Fill the blank: **{test_words[current_question][0]}**")
    run_test = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    expected = test_words[current_question][1]

    if run_test:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        result = "Show the correct sign..."

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

            if imgCrop.size > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                imgWhite = imgWhite.astype('float32') / 255.0
                imgWhite = cv2.resize(imgWhite, (224, 224))
                imgWhite = np.expand_dims(imgWhite, axis=0)
                prediction = model.predict(imgWhite)
                index = np.argmax(prediction)
                predicted = labels[index].strip().upper()

                if predicted == expected:
                    result = f"✅ Correct! It was {predicted}"
                    st.session_state["score"] = score + 1
                    st.session_state["current_question"] = current_question + 1
                else:
                    result = f"❌ You showed {predicted}, try again!"

        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(imgRGB)
        st.markdown(f"**{result}**")

    if current_question >= len(test_words):
        st.success(f"Test Finished! Your score: {score}/{len(test_words)}")
        if st.button("Restart"):
            st.session_state["current_question"] = 0
            st.session_state["score"] = 0
