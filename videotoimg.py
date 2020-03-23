import cv2

cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    # save frame as JPG file and change the path of the image based on wherever you want the image to be stored
    cv2.imwrite("training_images/0/frame%d.jpg" % count, test_img)
    count += 1
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ', resized_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
