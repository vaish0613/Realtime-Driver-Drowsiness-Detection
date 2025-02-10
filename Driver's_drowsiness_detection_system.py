from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import numpy as np
import dlib
import cv2
import imutils

# mixer for beep sound
mixer.init()
mixer.music.load("music.wav")


EAR_THRESH = 0.18  # Adjusted for better sensitivity
MAR_THRESH = 0.70  # Adjusted for accurate yawning detection
FRAME_CHECK = 25   # Increased to ensure prolonged inactivity triggers alarm

# Load face detection and facial landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  
    B = distance.euclidean(mouth[4], mouth[8])   
    C = distance.euclidean(mouth[0], mouth[6])   
    return (A + B) / (2.0 * C)

def adjust_contrast(frame, light_level):
    if light_level <= 4:
        alpha = 2.5 
        beta = 50    
    elif 5 <= light_level <= 7:
        alpha = 1.5 
        beta = 20    
    else:
        alpha = 1.0   
        beta = 0     
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def calculate_brightness(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(grayscale)

def map_brightness_to_light_level(brightness):
    return int((brightness / 255) * 9) + 1

cap = cv2.VideoCapture(0)
flag = 0
sleep = 0
drowsy = 0
active = 0
yawn = 0
drowsy_points = 0 
status = ""
color = (0, 0, 0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = calculate_brightness(frame)
    light_level = map_brightness_to_light_level(brightness)
    frame_contrast = adjust_contrast(frame, light_level)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]

        mouth = landmarks[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame_contrast, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_contrast, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_contrast, [mouthHull], -1, (0, 255, 0), 1)

        for (x, y) in landmarks:
            cv2.circle(frame_contrast, (x, y), 1, (255, 255, 255), -1)

        if ear < EAR_THRESH or mar > MAR_THRESH:
            flag += 1
            sleep += 1
            drowsy += 1
            active = 0
            if mar > MAR_THRESH:
                status = "Yawning!"
                color = (0, 0, 255)
            elif flag >= FRAME_CHECK or sleep > 8:  # Adjusted sleeping threshold
                drowsy_points += 1  
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                cv2.putText(frame_contrast, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
        elif EAR_THRESH <= ear < (EAR_THRESH + 0.05):
            flag = 0
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 4:  # Adjusted drowsy threshold
                status = "Drowsy Stage"
                color = (0, 255, 255)
        elif ear >= (EAR_THRESH + 0.05):
            flag = 0
            drowsy = 0  
            sleep = 0
            active += 1
            if active > 8:  # Adjusted active threshold
                status = "Active :)"
                color = (0, 255, 0)
                mixer.music.stop()

        cv2.putText(frame_contrast, f"EAR: {ear:.2f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_contrast, f"MAR: {mar:.2f}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_contrast, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame_contrast, f"Drowsy Points: {drowsy_points}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame_contrast)
    key = cv2.waitKey(1)
    if key == 27:  # Exit on pressing ESC
        break

cap.release()
cv2.destroyAllWindows()
