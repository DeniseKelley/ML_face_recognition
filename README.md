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
- [x] Identify the facial expression
- [ ] Replace facial expression with the appropriate emoji


## 1. Recognise the face using openCV

## Face detection using webcam with OpenCV and deep learning. 

[**Tutorial by Adrian Rosebrock**](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) was used to complete this part of the project. In order to perform fast and accurate face detection with OpenCV a pre-trained deep learniong Caffe model was applied. 

The first step is to load the model using file paths for [**Deep Neural Networks samples (dnn)**](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) for Caffe-based face detector: 

*    The __.prototxt__ file(s) which define the model architecture (i.e., the layers themselves)
*    The __.caffemodel__ file which contains the weights for the actual layers

When image is loaded the blob is created using dnn.blobFromImage that takes care of pre-processing of dimensions. To detect faces the blob is passed through the loaded model (```net```). Then the program loops over the detections and displays the confidence with boxes around faces using confidence threshold of 0.4 that filters out weak detections.  

![ezgif com-gif-maker(22)](https://user-images.githubusercontent.com/66845312/159154529-ff404036-0efb-4668-ad7f-2a36a0e4c26b.gif)


## 2. Identify the facial expression

Description in progress...

![ezgif com-gif-maker(24)](https://user-images.githubusercontent.com/66845312/165679564-e77bb841-a602-4e9f-a4e0-db845a70ab16.gif)


## 3. Replace facial expression with the appropriate emoji
