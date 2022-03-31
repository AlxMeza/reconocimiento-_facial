import cv2
import os
import numpy as np

dataPath = 'D:/Documentos/reconocimiento/facedata'
peoplelist = os.listdir(dataPath)
print('Lista de personas', peoplelist)

labels = []
facesdata = []

label = 0

for nameDir in peoplelist: 
    personPath = dataPath + '/' + nameDir
    print ('Leyendo Imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesdata.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
        #cv2.imshow('imagen', image)
        #cv2.waitKey(10)
    label = label + 1

    #print('Rotros de Alex: ', np.count_nonzero(np.array(labels)==0))
    #print('Rotros de Tobey: ', np.count_nonzero(np.array(labels)==1))

    """EigenFaces"""
    # face_recognizer = cv2.face.EigenFaceRecognizer_create()

    # #Entrenando el reconocedor de Rostros
    # print("Entrenando...")
    # face_recognizer.train(facesdata, np.array(labels))

    # #Almacenando el modelo obtenido
    # face_recognizer.write('modeloEigenFace.xml')
    # print("El modelo fue almacenado")

    """Fisher Face"""
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()

    # #Entrenando el reconocedor de Rostros
    # print("Entrenando...")
    # face_recognizer.train(facesdata, np.array(labels))

    # #Almacenando el modelo obtenido
    # face_recognizer.write('modeloFisherFace.xml')
    # print("El modelo fue almacenado")

    """LBPH"""
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #Entrenando el reconocedor de Rostros
    print("Entrenando...")
    face_recognizer.train(facesdata, np.array(labels))

    #Almacenando el modelo obtenido
    face_recognizer.write('modeloLBPH.xml')
    print("El modelo fue almacenado")
