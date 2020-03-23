from cv2 import cv2
import os
import numpy as np
import faceRecognition as fr

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("td.yml")

name = {0: "person 1", 1: "person 2"}
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    face_detected, gray_img = fr.faceDetection(test_img)

    for (x, y, w, h) in face_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)

    for face in face_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(
            roi_gray)  # predicting the label of given image
        print("confidence:", confidence)
        print("label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if confidence < 39:  # If confidence less than 37 then don't print predicted face text on screen
            fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition tutorial ', resized_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
