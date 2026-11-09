from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)

# Global variables
encode_list_known = []
class_names = []
last_logged_time = {}
cap = None

# Load encodings
def load_encodings():
    global encode_list_known, class_names
    try:
        with open('encoding_file.pkl', 'rb') as file:
            data = pickle.load(file)
            encode_list_known = data['encodings']
            class_names = data['names']
        print(f"Loaded {len(class_names)} faces")
    except:
        print("Encoding file not found!")

load_encodings()

# Generate video frames
def generate_frames():
    global cap, last_logged_time
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        
        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encode_list_known, encodeFace)
            faceDis = face_recognition.face_distance(encode_list_known, encodeFace)
            matchIndex = faceDis.argmin()
            
            if matches[matchIndex]:
                name = class_names[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                # Mark attendance
                mark_attendance(name)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def mark_attendance(name):
    global last_logged_time
    t = datetime.now().strftime("%H:%M:%S")
    d = datetime.now().strftime("%Y-%m-%d")
    
    if name not in last_logged_time or (time.time() - last_logged_time[name]) > 60:
        filename = 'attendance.csv'
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
            df.to_csv(filename, index=False)
        
        new_entry = pd.DataFrame({'Name': [name], 'Time': [t], 'Date': [d]})
        new_entry.to_csv(filename, mode='a', header=False, index=False)
        last_logged_time[name] = time.time()
        print(f"Marked: {name}")

# Routes
@app.route('/')
def index():
    return render_template('index.html', names=class_names)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    try:
        df = pd.read_csv('attendance.csv')
        data = df.to_dict('records')
        return render_template('attendance.html', data=data)
    except:
        return render_template('attendance.html', data=[])

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    #http://localhost:5000