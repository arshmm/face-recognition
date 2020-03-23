from cv2 import cv2
import os
import numpy as np
import faceRecognition as fr

test_img_path = "test_images/img.jpg"
test_img = cv2.imread(test_img_path)
face_detected, gray_img = fr.faceDetection(test_img)

print("faces detected", face_detected)

# this is for initial training the machine
faces, faceID = fr.lables_for_train("training_images")
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save("td.yml")

# once the training is done and the yaml file is created use this so you dont have to perform training everytime you try detection
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# ace_recognizer.read("td.yml")

# if you using this multiple face, add the names here
name = {0: "person1" ''',1: "person 2"'''}

for face in face_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y+w, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence", confidence)
    print("lable", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if(confidence > 25):
        continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow("detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
