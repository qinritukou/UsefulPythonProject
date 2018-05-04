import cv2 

# Loading the cascades 
face_cascode = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascode = cv2.CascadeClassifier("haarcascade_eye.xml")

# Defining a fucntion that will do the dections
def detect(gray, frame):
    faces = face_cascode.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascode.detectMultiScale(roi_gray, 1.3, 5)
        for (roi_x, roi_y, roi_w, roi_h) in eyes:
            eye_x = x + roi_x
            eye_y = y + roi_y
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + roi_w, eye_y + roi_h), (0, 255, 0), 2)
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