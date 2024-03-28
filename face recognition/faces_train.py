import os
import cv2 as cv
import numpy as np

people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]

pathway_to_face_train_images = r'D:\Image_Processing_Section\FreeCodeAcademy\FACEs\Faces\train'
DIR = pathway_to_face_train_images

haar_cascade = cv.CascadeClassifier("haar_face.xml")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x: x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("TRAINING DONE ---------------------------")


## To be sure that our function works properly...
# print(f'Lenght of the features : {len(features)}')
# print(f'Lenght of the labels : {len(labels)}')



# TRAINING PART

features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face_LBPHFaceRecognizer.create()

# train the recognizer on the features list and the label list

face_recognizer.train(features, labels)

face_recognizer.save("face_trained.yml")
np.save("features.npy", features)
np.save("labels.npy", labels)
