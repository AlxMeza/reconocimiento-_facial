import cv2
import os
import imutils

#personName = 'Alex'
personName = 'Tobey'
dataPath = 'D:/Documentos/reconocimiento/facedata'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta Creada: ', personPath)
    os.makedirs(personPath)

#cap = cv2.VideoCapture('Alex1.mp4')
cap = cv2.VideoCapture('Tobey2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 300

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxframe = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.1, 8)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        rostro = auxframe[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150))
        cv2.imwrite(personPath +'/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 500:
        break

cap.release()
cv2.destroyAllWindows()