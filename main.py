import cv2
import numpy as np
import face_recognitions # type: ignore

imgModi = face_recognitions.load_image_file('Images_Attendance/modi-image-for-InUth.jpg')
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)
imgTest = face_recognitions.load_image_file('Images_Attendance/narendra-modi.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognitions.face_locations(imgModi)[0]
encodeModi = face_recognitions.face_encodings(imgModi)[0]
cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

facelocTest = face_recognitions.face_locations(imgTest)[0]
encodeTest = face_recognitions.face_encodings(imgTest)[1]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

results = face_recognitions.compare_faces([encodeModi], encodeTest)
faceDis = face_recognitions.face_distance([encodeModi], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

cv2.imshow('modi', imgModi)
cv2.imshow('narendra-modi', imgTest)
cv2.waitKeys(0)