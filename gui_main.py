import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import time
from PIL import Image, ImageTk
from threading import Thread

class FaceAttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2C3E50")

        # Variables
        self.is_running = False
        self.cap = None
        self.encode_list_known = []
        self.class_names = []
        self.last_logged_time = {}
        
        # Load encodings on startup
        self.load_encodings()

        # GUI Layout
        self.create_gui()

    def load_encodings(self):
        """Load pre-saved face encodings"""
        try:
            with open('encoding_file.pkl', 'rb') as file:
                data = pickle.load(file)
                self.encode_list_known = data['encodings']
                self.class_names = data['names']
            print(f"Loaded {len(self.class_names)} faces")
        except Exception as e:
            messagebox.showwarning("Warning", f"Encodings file not found! Run encode_faces.py first.\nError: {e}")

    def create_gui(self):
        """Create the GUI interface"""
        
        # Title
        title_label = tk.Label(self.root, text="Face Recognition Attendance System", 
                              font=("Helvetica", 24, "bold"), bg="#2C3E50", fg="white")
        title_label.pack(pady=10)

        # Main Frame (Left: Video, Right: Controls)
        main_frame = tk.Frame(self.root, bg="#2C3E50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Frame - Video Display
        self.video_frame = tk.Frame(main_frame, bg="black", width=640, height=480)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame, text="Camera Feed", 
                                   bg="black", fg="white", font=("Helvetica", 14))
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right Frame - Controls
        control_frame = tk.Frame(main_frame, bg="#34495E", width=300, height=480)
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Control Frame Title
        ctrl_title = tk.Label(control_frame, text="Controls", 
                              font=("Helvetica", 16, "bold"), bg="#34495E", fg="white")
        ctrl_title.pack(pady=15)

        # Start Button
        self.start_btn = tk.Button(control_frame, text="▶ START CAMERA", 
                                    font=("Helvetica", 12, "bold"), bg="#27AE60", fg="white",
                                    command=self.start_camera, width=20, height=2)
        self.start_btn.pack(pady=10)

        # Stop Button
        self.stop_btn = tk.Button(control_frame, text="⏹ STOP CAMERA", 
                                   font=("Helvetica", 12, "bold"), bg="#E74C3C", fg="white",
                                   command=self.stop_camera, width=20, height=2)
        self.stop_btn.pack(pady=10)

        # Encode Faces Button
        self.encode_btn = tk.Button(control_frame, text="🔄 RE-ENCODE FACES", 
                                   font=("Helvetica", 12, "bold"), bg="#F39C12", fg="white",
                                   command=self.run_encoder, width=20, height=2)
        self.encode_btn.pack(pady=10)

        # View Attendance Button
        self.view_btn = tk.Button(control_frame, text="📊 VIEW ATTENDANCE", 
                                   font=("Helvetica", 12, "bold"), bg="#3498DB", fg="white",
                                   command=self.view_attendance, width=20, height=2)
        self.view_btn.pack(pady=10)

        # Quit Button
        self.quit_btn = tk.Button(control_frame, text="❌ QUIT", 
                                  font=("Helvetica", 12, "bold"), bg="#7F8C8D", fg="white",
                                  command=self.quit_app, width=20, height=2)
        self.quit_btn.pack(pady=10)

        # Status Bar
        self.status_label = tk.Label(self.root, text="Status: Ready", 
                                      font=("Helvetica", 12), bg="#2C3E50", fg="#2ECC71")
        self.status_label.pack(pady=5)

        # Info Label
        info_label = tk.Label(self.root, 
                              text=f"Known Faces: {len(self.class_names)} | {', '.join(self.class_names)}", 
                              font=("Helvetica", 10), bg="#2C3E50", fg="#BDC3C7")
        info_label.pack(pady=5)

    def start_camera(self):
        """Start the webcam"""
        if self.is_running:
            messagebox.showinfo("Info", "Camera is already running!")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return

        self.is_running = True
        self.status_label.config(text="Status: Camera Running", fg="#2ECC71")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Start video loop in a separate thread
        self.video_thread = Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

    def stop_camera(self):
        """Stop the webcam"""
        self.is_running = False
        self.status_label.config(text="Status: Camera Stopped", fg="#E74C3C")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset video label
        self.video_label.config(text="Camera Stopped", bg="black")

    def video_loop(self):
        """Main video processing loop"""
        while self.is_running:
            success, img = self.cap.read()
            if not success:
                break

            # Resize for faster processing
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces
            faceCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            # Loop through detected faces
            for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(self.encode_list_known, encodeFace)
                faceDis = face_recognition.face_distance(self.encode_list_known, encodeFace)
                
                matchIndex = faceDis.argmin()
                
                if matches[matchIndex]:
                    name = self.class_names[matchIndex].upper()
                    
                    # Draw box
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    # Mark attendance
                    self.mark_attendance(name)

            # Convert to display on GUI
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB = cv2.resize(imgRGB, (640, 480))
            img_pil = Image.fromarray(imgRGB)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.video_label.config(image=img_tk, text="")
            self.video_label.image = img_tk  # Keep reference

        # Release camera when loop ends
        if self.cap:
            self.cap.release()

    def mark_attendance(self, name):
        """Mark attendance in CSV"""
        current_time = datetime.now()
        t = current_time.strftime("%H:%M:%S")
        d = current_time.strftime("%Y-%m-%d")

        # Check cooldown (60 seconds)
        if name not in self.last_logged_time or (time.time() - self.last_logged_time[name]) > 60:
            filename = 'attendance.csv'
            try:
                df = pd.read_csv(filename)
            except:
                df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
                df.to_csv(filename, index=False)
            
            new_entry = pd.DataFrame({'Name': [name], 'Time': [t], 'Date': [d]})
            new_entry.to_csv(filename, mode='a', header=False, index=False)
            
            self.last_logged_time[name] = time.time()
            self.status_label.config(text=f"Attendance Marked: {name}", fg="#F39C12")
            print(f"Marked: {name} at {t}")

    def run_encoder(self):
        """Run the encoder script"""
        messagebox.showinfo("Info", "Running encoder... Check terminal for progress.")
        import encode_faces
        self.load_encodings()
        messagebox.showinfo("Success", "Encoding complete! Loaded new faces.")
        info_label = tk.Label(self.root, 
                              text=f"Known Faces: {len(self.class_names)} | {', '.join(self.class_names)}", 
                              font=("Helvetica", 10), bg="#2C3E50", fg="#BDC3C7")
        info_label.pack(pady=5)

    def view_attendance(self):
        """Open attendance CSV"""
        try:
            os.startfile('attendance.csv')  # Windows
        except:
            messagebox.showinfo("Info", "Open 'attendance.csv' file manually to view.")

    def quit_app(self):
        """Quit the application"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceGUI(root)
    root.mainloop()