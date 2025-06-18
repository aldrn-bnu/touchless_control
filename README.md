📄 README.md for working_code.txt (Early Version)

# ✋ Touchless Control – Initial Version

This is the **initial working prototype** of the gesture-based touchless control system, later refined and uploaded as [`touchless_control_main`](https://github.com/aldrn-bnu/touchless_control_main) in this profile.

## 🔧 Overview

A Python OpenCV + MediaPipe-powered gesture interface that lets you:
- Move the mouse cursor with your hand
- Draw on a virtual whiteboard
- Scroll content by finger gestures
- Use system-level clipboard functions like **copy** and **paste**
- Toggle drawing canvas visibility using a pinky gesture

---

## 🧠 Features

### ✍️ Gesture-Based Controls

- **Pinch (thumb + index):** Left click / draw
- **Index + Middle fingers up:** Erase or Scroll (based on movement)
- **Index + Middle + Ring up:** Copy (`Ctrl + C`)
- **All fingers up:** Paste (`Ctrl + V`)
- **Only pinky up:** Toggle whiteboard (hold 1s)

### 🎨 Whiteboard Capabilities

- Smooth drawing with finger strokes
- Real-time feedback with overlays
- Optional erasing and clearing (`c` key)

### 🖱️ Mouse Interaction

- Control cursor with your hand
- Smooth movement via Kalman-like filtering
- Handles screen boundaries and positioning

---

## 📦 Requirements

See `requirements.txt`, but core libraries include:
- OpenCV
- MediaPipe
- PyAutoGUI
- pyperclip
- win32api (for Windows-based input simulation)

---

## 🚀 How to Run


python working_code.txt
Ensure your webcam is connected and pointing at your hand in a well-lit environment.

📘 Developer Notes
Developed using MediaPipe Hands for landmark detection

This version includes foundational features later improved in rith.py and frontend UI single.py

Refer to the touchless_control_main repo for the latest version with:

Virtual monitor support

Streamlit dashboard UI

Lottie animations

Team profile & tutorial section

👨‍💻 Author
This project is created by Aldrin Binu and team as part of a gesture-control research and application initiative.

📦 Final Version: touchless_control_main

📜 License
MIT License – Free to use, modify, and redistribute.

---

