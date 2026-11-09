import cv2
import face_recognition
import os
import pickle

# Path to the folder containing known faces
path = 'images'

# Lists to store encodings and names
images = []
classNames = []
myList = os.listdir(path)
print(f"Found {len(myList)} images in the folder.")

# Loop through the images
for cl in myList:
    # Read the image
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    
    # Get the name from the filename (remove extension)
    classNames.append(os.path.splitext(cl)[0])

print("Encoding faces...")

# Function to find encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert BGR (OpenCV) to RGB (face_recognition)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encode the face
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Generate the encodings
encodeListKnown = findEncodings(images)

# Save the encodings and names to a file so we don't have to recalculate them
encodeListKnownWithNames = {'encodings': encodeListKnown, 'names': classNames}
with open('encoding_file.pkl', 'wb') as file:
    pickle.dump(encodeListKnownWithNames, file)

print("Encoding Complete! You can now run main.py")