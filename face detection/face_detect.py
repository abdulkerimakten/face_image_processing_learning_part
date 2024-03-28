import cv2 as cv

img = cv.imread("Photos/group 1.jpg")
cv.imshow("Original Image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)


# Face Detection
haar_cascade = cv.CascadeClassifier("haar_face.xml")
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
"""
The 'detectMultiScale()' method returns a list of rectangles where objects (faces) are detected. 
Each rectangle corresponds to a detected object and represents its bounding box. 
The coordinates of the bounding box (x, y, width, height) can be used to draw rectangles around the detected objects in the image.
"""

print(f"Number of faces detected : {len(faces_rect)}")


## drawing a rectangle over a detected face

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (5,255,5), thickness=2)

cv.imshow("Face Detected", img)


cv.waitKey(0)