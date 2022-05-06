# Emojify - Replace a Face in Real Time With Emoji
The final product of this project will replace a face with an appropriate emoji by using *computer vision* and *deep learning*. 

## The following emotions to be detected

* Angry
* Disgust
* Fear 
* Happy 
* Sad 
* Surprise  
* Neutral

## Tasks 

- [x] Recognise the face using openCV
- [x] Identify the facial expression
- [x] Replace facial expression with the appropriate emoji


## 1. Recognise the face using openCV

## Face detection using webcam with OpenCV and deep learning. 

[**Tutorial by Adrian Rosebrock**](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) was used to complete this part of the project. In order to perform fast and accurate face detection with OpenCV a pre-trained deep learniong Caffe model was applied. 

The first step is to load the model using file paths for [**Deep Neural Networks samples (dnn)**](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) for Caffe-based face detector: 

*    The __.prototxt__ file(s) which define the model architecture (i.e., the layers themselves)
*    The __.caffemodel__ file which contains the weights for the actual layers

When image is loaded the blob is created using dnn.blobFromImage that takes care of pre-processing of dimensions. To detect faces the blob is passed through the loaded model (```net```). Then the program loops over the detections and displays the confidence with boxes around faces using confidence threshold of 0.4 that filters out weak detections.  

![ezgif com-gif-maker(22)](https://user-images.githubusercontent.com/66845312/159154529-ff404036-0efb-4668-ad7f-2a36a0e4c26b.gif)


## 2. Identify the facial expression
 
For this part several models were trained and tested. Model created by Anshal Singh has accuracy of 0.62 and model created by Gaurav Sharma has accuracy of 0.72. Even though the accuracy data shows that Sharma's model is more accurate, on practice it had difficulty recognising most of the emotions. Thus Singh's odel was used for the rest of tthis project.

According to the data count both models used the same data set. There are 7 categories of emotions in this data set. Emotion happiness has the largest amount of images and emotion disgust has the minimum images. Through visualization the data was inspected: 

    * The data contains images of people from different genders, races, and ages
    * The data contains non human images, like cartoons
    * The data contains images with different trimms and agianst different backgrounds



![ezgif com-gif-maker(30)](https://user-images.githubusercontent.com/66845312/167151803-55b79d87-fda7-4737-b5ba-a783fc001b9c.gif)


## 3. Replace facial expression with the appropriate emoji

For the final part emojies were downloaded from web and each nedded to be reshaped to match the dimensions of the captured face to replace it. 


![ezgif com-gif-maker(29)](https://user-images.githubusercontent.com/66845312/167152089-93459f1d-8e42-42e5-8748-95f3bce12ebc.gif)
