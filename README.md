# Emojify - Replace a Face in Real Time With Emoji
The final product of this project will replace a face with an appropriate emoji by using *computer vision* and *deep learning*. 

## The following emotions to be detected

    * angry
    * disgust
    * happy
    * sad
    * surprise
    * natural

## Tasks 

- [x] Recognise the face using openCV
- [ ] Identify the facial expression
- [ ] Replace facial expression with the appropriate emoji


## 1. Recognise the face using openCV

**Face detection using webcam with OpenCV and deep learning.** 
[Tutorial by Adrian Rosebrock] (https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ "Face detection") was used to complete this part of the project. In order to perform fast and accurate face detection with OpenCV a pre-trained deep learniong Caffe model was applied. 

[**Deep Neural Networks samples (dnn)**] (https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector "dnn") for Caffe-based face detector include two files: 

    *    The *.prototxt* file(s) which define the model architecture (i.e., the layers themselves)
    *    The *.caffemodel* file which contains the weights for the actual layers




## 2. Identify the facial expression
## 3. Replace facial expression with the appropriate emoji