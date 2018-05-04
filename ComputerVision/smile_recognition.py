import cv2 

# Loading the cascades 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Defining a fucntion that will do the dections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (roi_x, roi_y, roi_w, roi_h) in eyes:
            eye_x = x + roi_x
            eye_y = y + roi_y
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + roi_w, eye_y + roi_h), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (roi_x, roi_y, roi_w, roi_h) in smiles:
            smile_x = x + roi_x
            smile_y = y + roi_y
            cv2.rectangle(frame, (smile_x, smile_y), (smile_x + roi_w, smile_y + roi_h), (0, 0, 255), 2)


    return frame

# Doing some Face Recognition with the webcam 
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()