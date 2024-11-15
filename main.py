import cv2
import random
import time
from deepface import DeepFace
from collections import Counter

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

target_emotion = random.choice(emotions_list)

emotion_start_time = None
expression_duration = 3  

emotion_buffer = []
buffer_size = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_emotion = None

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_roi = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]['dominant_emotion']
            emotion_buffer.append(current_emotion)

            if len(emotion_buffer) > buffer_size:
                emotion_buffer.pop(0)

            most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]

            if most_common_emotion == target_emotion:
                if emotion_start_time is None:
                    emotion_start_time = time.time() 
                if time.time() - emotion_start_time >= expression_duration:
                    target_emotion = random.choice(emotions_list)
                    emotion_start_time = None 
            else:
                emotion_start_time = None

        except Exception as e:
            print(f"Error: {e}")

    cv2.putText(frame, f"Express: {target_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(frame, f"Detected: {most_common_emotion if current_emotion else 'None'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
