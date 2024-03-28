# Face Detection and Recognition Application

This application performs face detection and recognition using OpenCV and Haar Cascade classifiers. Face detection uses the Haar Cascade method to identify face regions in an image, while face recognition allows for the identification of faces within the detected face regions.

**Warning:** Users may need to adjust file paths according to their own systems when downloading the repository files. Otherwise, the code may not work properly.


### Installation
This application is written in Python and requires the OpenCV library. You can install OpenCV using the following command:

```bash
pip install opencv-python
````
## Haar Cascade Model

Face detection and recognition rely on Haar Cascade classifiers. These classifiers are stored in .xml files and should be included in the project.

## Running the Application

You can start the application by running the proper Python file or you can see the outputs in an IDE. This script captures images from either a camera or a video file, detects faces, and recognizes them.
## Results

The application visualizes the results by drawing rectangles around the detected faces on the screen.

## Notes

- You can change or customize the Haar Cascade model by using different .xml files. Experimenting with different models may yield better detection and recognition results.

- The application can capture images from various sources such as video files or live camera streams. You can modify the image source by adjusting the parameters of the `cv.VideoCapture()` function.

