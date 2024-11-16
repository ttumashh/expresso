# Expresso Emotion Detection

## Overview
This project is an interactive emotion detection game that uses **OpenCV** for real-time webcam input and **DeepFace** for facial emotion analysis. The program randomly selects a target emotion and challenges the user to mimic that emotion for a specified duration.

---

## Features
- **Real-Time Emotion Detection**: 
  - Captures live video from the webcam.
  - Detects faces and analyzes the dominant emotion using DeepFace.
- **Interactive Gameplay**:
  - Randomly selects a target emotion from the following: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.
  - Detects and tracks if the user matches the target emotion.
- **Dynamic Visual Feedback**:
  - Displays the target emotion and most recently detected emotion on the screen.
  - Highlights detected faces with bounding boxes.

---

## Prerequisites

### Required Libraries
- `opencv-python`
- `deepface`
- `collections`
- `random`
- `time`
- `tf-keras`

### Installation and Run
Install the necessary libraries with:
```bash
pip install -r requirements.txt
python main.py or python3 main.py
```

### Controls
- q: quit the game
