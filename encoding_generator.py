import os
import cv2
import face_recognition
import pickle

#  importing the student images
folderPath = "images"
Pathlist = os.listdir(folderPath)
imageList = []
studentIds = []

for path in Pathlist:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is None:
        print(f"Warning: Could not read image {path}")
        continue
    imageList.append(img)
    studentIds.append(os.path.splitext(path)[0])

print(studentIds)

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face found in image.")
    return encodeList

print("Encoding started")
encodeListKnown = findEncodings(imageList)
print("Encoding Complete")

file = open("encoding.p", "wb")
pickle.dump([encodeListKnown, studentIds], file)
file.close()
print("Encoding file saved")