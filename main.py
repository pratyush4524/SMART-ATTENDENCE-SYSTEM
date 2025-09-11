import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone


cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

imgBackground = cv2.imread("resources/background.png")

folderModePath = "resources/Modes"
modePathlist = os.listdir(folderModePath)
imageModeList = []
print("Available modes:", modePathlist)

for i in range(0, 5):
    imageModeList.append(cv2.imread(f"resources/Modes/{i}.png"))

print("loding encode files...")
file = open("encoding.p", "rb")
encodeListKnownwithides = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownwithides
print("Encode file loaded")

modeType = 0

while True:
    success, img = cap.read()

    imgs = cv2.resize(img,(0, 0),None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imageModeList[1]

    for encodeFace, faceloc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("Matches:", matches)
        print("Face Distance:", faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = faceloc
            y1 = int(y1 * 4)
            x2 = int(x2 * 4)
            y2 = int(y2 * 4)
            x1 = int(x1 * 4)

            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground,bbox, rt=0)

   # cv2.imshow("webcam", img)
    cv2.imshow("background", imgBackground)
    cv2.waitKey(1)