import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier("haar_face.xml")

people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]

# features = np.load("features.npy")
# labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read(r"face recognition\face_trained.yml")

# Here you can try different images to recognize
img = cv.imread(r"Faces\val\madonna\3.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)


# Detect the face in the image
face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# Face recognition and labeling
for (x, y, w, h) in face_rect:
    favce_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(favce_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    """
    Understanding Confidence Values:
    The confidence value represents the dissimilarity or error between the detected face and the recognized face. 
    that a lower confidence value is typically better because it means that the detected face is more similar to the recognized face.
    
    Also, the confidence value is not expressed as a percentage!
    
    """

    cv.putText(img, str(people[label]), [20,20], cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("DETECTED FACE", img)

cv.waitKey(0)