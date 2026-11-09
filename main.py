import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import time

# Load the pre-saved encodings
print("Loading encodings...")
with open('encoding_file.pkl', 'rb') as file:
    data = pickle.load(file)
    encodeListKnown = data['encodings']
    classNames = data['names']

print("Loaded known faces:", classNames)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create attendance CSV if it doesn't exist
filename = 'attendance.csv'
try:
    df = pd.read_csv(filename)
except:
    df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
    df.to_csv(filename, index=False)

# To prevent spamming the log, we track if a person was marked recently
last_logged_time = {} 

while True:
    success, img = cap.read()
    
    # Resize image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Loop through detected faces
    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        # Compare with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Find the best match (lowest distance)
        matchIndex = faceDis.argmin()
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            # Draw a bounding box around the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # MARK ATTENDANCE
            current_time = datetime.now()
            t = current_time.strftime("%H:%M:%S")
            d = current_time.strftime("%Y-%m-%d")

            # Logic: Only log if they haven't been logged in the last 60 seconds
            if name not in last_logged_time or (time.time() - last_logged_time[name]) > 60:
                # Save to CSV
                new_entry = pd.DataFrame({'Name': [name] , 'Time': [t] , 'Date': [d] })
                new_entry.to_csv(filename, mode='a', header=False, index=False)
                
                last_logged_time[name] = time.time()
                print(f"Attendance Marked: {name} at {t}")

    # Show the video
    cv2.imshow('Webcam', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()