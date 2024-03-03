import cv2
import os
from keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound

model = load_model("drowsiness_model3.h5")

face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
leye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0

rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)

        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(64, 64)
        r_eye = np.expand_dims(r_eye, axis=-1)
        r_eye = np.repeat(r_eye, 3, axis=-1)
        r_eye = np.expand_dims(r_eye, axis=0)

        r_eye = r_eye.astype('float32')
        rpred = model.predict(r_eye)

        if rpred[0] > 0.5:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)# Green rectangle for open eyes
        else:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)# Red rectangle for closed eyes

        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)

        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(64, 64)
        l_eye = np.expand_dims(l_eye, axis=-1)
        l_eye = np.repeat(l_eye, 3, axis=-1)
        l_eye = np.expand_dims(l_eye, axis=0)

        l_eye = l_eye.astype('float32')
        lpred = model.predict(l_eye)

        if lpred[0] > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green rectangle for open eyes
        else:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)# Red rectangle for closed eyes

        break

    if rpred[0] > 0.5 and lpred[0] > 0.5:
        score = score + 1
        cv2.putText(frame, "Eye Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if score>=1:
            cv2.putText(frame, "Alarm will be ON", (10, height-100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if score >= 5:
            
            playsound("alarm.wav")
        
    else:
        score = score - 1
        cv2.putText(frame, "Eye Opened", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 