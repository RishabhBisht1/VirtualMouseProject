# 🖱️ Virtual Mouse using Hand Tracking

This project allows you to control your mouse using hand gestures captured from your webcam, utilizing computer vision and hand tracking.

## 📌 Features

- Move mouse pointer with your index finger.
- Click using gesture (index + middle finger pinch).
- Smooth pointer movement.
- Region boundary for better control.

## 📂 Project Structure
VirtualMouseProject/
│
├── HandTrackingModule.py # Contains hand detection logic using Mediapipe
├── VirtualMouse.py # Main script to run virtual mouse
└── README.md # This file

## 🧠 Tech Stack

- Python
- OpenCV
- Mediapipe
- NumPy
- pynput (for mouse control)
