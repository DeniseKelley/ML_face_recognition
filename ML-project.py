
import numpy as np
import os
import cv2 as cv

#set the path to main folder
main_folder_path = os.path.dirname(os.path.abspath(__file__)) + os.sep 
# objects' confidence threshold for example: threshold = 0.5 is considered to be high
threshold = 0.4  

#setting paths to Caffe prototxt file and Caffe model 
#two diffirent ways to set a path

# the path of main folder + the file
prototxt_file = main_folder_path + 'Resnet_SSD_deploy.prototxt' 
#direct path to the file
caffemodel_file = "/Users/annakelley/Desktop/ML_face_recognition/Res10_300x300_SSD_iter_140000.caffemodel" 
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file) #dnn - deep neural networks caffee is a deep learning library
print('ResNetSSD caffe model loaded successfully')

#open the camera 0 -is the index of the main camera
cap = cv.VideoCapture(0)

#########
# creating a video writer
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20  
videoW_f = cv.VideoWriter_fourcc(*'mp4v')  
writer = cv.VideoWriter()
#########

#save the video to the main folder test_out file
out_path = main_folder_path+'test_out'+os.sep+'example.mp4' #where I save the video
writer.open(out_path, videoW_f, out_fps, size, True)   #writing into that file

while True:
    #cam object reads the frame one by one, check is true when camera is open
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    check, image = cap.read()
    (h, w) = image.shape[:2]
    #giving the frame to the algorithm
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    #store the model as net 
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)   #blob - image is in the form desired by algo
    detections = net.forward() #gives the predictions
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
	    # prediction
        confidence = detections[0, 0, i, 2]

        # compare confidence to the confidence threshold of 0.4
        # filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        # less than the higher confidence => comtinue
        if confidence < threshold:
            continue
        
        # compute the (x, y)-coordinates of the bounding box for the
		# object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")

        # draw the bounding box of the face along with the associated
		# probability
        text = '{0:.2f}%'.format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv.putText(image, text, (startX, y),
		cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            


    #show the output frame        
    cv.imshow('Frame', image)
    writer.write(image) #writing into the file 

    key =cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#do a bit of cleanup
writer.release()
cap.release()
cv.destroyAllWindows()

