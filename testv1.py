import cv2
import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, Label, Button
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Constants
offset = 20
imgSize = 300

# Init camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Tkinter window setup
root = tk.Tk()
root.title("ASL Learning App (A-E)")
root.geometry("800x600")

# Global frame for switching
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# ---------- Helper Functions ----------
def switch_frame(frame_func):
    for widget in main_frame.winfo_children():
        widget.destroy()
    frame_func()

def back_and_exit_buttons():
    btn_frame = tk.Frame(main_frame)
    btn_frame.pack(pady=10)
    Button(btn_frame, text="Back to Menu", command=lambda: switch_frame(menu_page), width=15).pack(side="left", padx=10)
    Button(btn_frame, text="Exit", command=root.destroy, width=15).pack(side="left", padx=10)

# ---------- Learn Page ----------
def learn_page():
    Label(main_frame, text="Learn ASL (A-E)", font=("Arial", 24)).pack(pady=10)
    letter_var = tk.StringVar(value="A")

    def show_image():
        letter = letter_var.get()
        img = Image.open(f"images/{letter}.png")
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

    option = ttk.Combobox(main_frame, textvariable=letter_var, values=["A", "B", "C", "D", "E"])
    option.pack(pady=10)
    option.bind("<<ComboboxSelected>>", lambda e: show_image())

    img_label = Label(main_frame)
    img_label.pack()
    show_image()

    back_and_exit_buttons()

# ---------- Train Page ----------
def train_page():
    Label(main_frame, text="Train (Live Feedback)", font=("Arial", 24)).pack(pady=10)
    video_label = Label(main_frame)
    video_label.pack()

    result_var = tk.StringVar()
    Label(main_frame, textvariable=result_var, font=("Arial", 20)).pack(pady=10)

    running = True

    def show_frame():
        if not running:
            return

        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

            if imgCrop.size > 0:
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

                imgWhite = imgWhite.astype('float32') / 255.0
                imgWhite = cv2.resize(imgWhite, (224, 224))
                imgWhite = np.expand_dims(imgWhite, axis=0)
                prediction = model.predict(imgWhite)
                index = np.argmax(prediction)
                result_var.set(f"Prediction: {labels[index]}")

        img = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        video_label.configure(image=img)
        video_label.image = img
        video_label.after(10, show_frame)

    show_frame()
    back_and_exit_buttons()

# ---------- Test Page ----------
def test_page():
    Label(main_frame, text="Test Your ASL!", font=("Arial", 24)).pack(pady=10)

    video_label = tk.Label(main_frame)
    video_label.pack()

    question_label = Label(main_frame, font=("Arial", 20))
    question_label.pack(pady=10)

    result_var = tk.StringVar()
    Label(main_frame, textvariable=result_var, font=("Arial", 16)).pack()

    test_words = [("B_D", "E"), ("_AIT", "B"), ("_ook", "C"), ("_ATE", "D")]
    current = [0]
    score = [0]

    # Variables for hold detection
    hold_start_time = [None]
    last_predicted = [None]
    visual_feedback_shown = [False]

    def show_question():
        if current[0] >= len(test_words):
            result_var.set(f"Final Score: {score[0]}/{len(test_words)}")
            question_label.config(text="Test Completed!")
            return
        question_label.config(text=f"Fill the blank: {test_words[current[0]][0]}")

    def proceed_to_next_question():
        current[0] += 1
        hold_start_time[0] = None
        last_predicted[0] = None
        visual_feedback_shown[0] = False
        if current[0] < len(test_words):
            show_question()
        else:
            result_var.set(f"Final Score: {score[0]}/{len(test_words)}")
            question_label.config(text="Test Completed!")

    def check_prediction():
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        predicted = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

            if imgCrop.size > 0:
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

                imgWhite = imgWhite.astype('float32') / 255.0
                imgWhite = cv2.resize(imgWhite, (224, 224))
                imgWhite = np.expand_dims(imgWhite, axis=0)
                prediction = model.predict(imgWhite)
                index = np.argmax(prediction)
                predicted = labels[index].strip().upper()

        expected = test_words[current[0]][1].upper() if current[0] < len(test_words) else None
        current_time = time.time()

        if predicted == expected:
            if last_predicted[0] != predicted:
                hold_start_time[0] = current_time
                last_predicted[0] = predicted
                result_var.set(f"Hold '{predicted}' steady...")
            else:
                if hold_start_time[0] and (current_time - hold_start_time[0]) >= 3 and not visual_feedback_shown[0]:
                    result_var.set("✅ Correct! Moving to next...")
                    visual_feedback_shown[0] = True
                    score[0] += 1
                    root.after(2000, proceed_to_next_question)
        else:
            hold_start_time[0] = None
            last_predicted[0] = None
            visual_feedback_shown[0] = False
            if predicted is not None:
                result_var.set(f"❌ Try Again (You showed: {predicted})")

        img = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        video_label.configure(image=img)
        video_label.image = img
        if current[0] < len(test_words):
            video_label.after(100, check_prediction)

    show_question()
    check_prediction()
    back_and_exit_buttons()

# ---------- Main Menu ----------
def menu_page():
    Label(main_frame, text="ASL Learning App (A-E)", font=("Arial", 28)).pack(pady=20)
    Button(main_frame, text="Learn", command=lambda: switch_frame(learn_page), width=20, height=2).pack(pady=10)
    Button(main_frame, text="Train", command=lambda: switch_frame(train_page), width=20, height=2).pack(pady=10)
    Button(main_frame, text="Test", command=lambda: switch_frame(test_page), width=20, height=2).pack(pady=10)
    Button(main_frame, text="Exit", command=root.destroy, width=20, height=2).pack(pady=10)

# Launch the app
switch_frame(menu_page)
root.mainloop()
cap.release()
cv2.destroyAllWindows()
