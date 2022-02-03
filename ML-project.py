import cv2 


cam = cv2.VideoCapture(0) ##open the camera 0-is the index of camera(main camera)

while True:
    check, frame = cam.read()  #cam object reads the frame one by one, check is true when camera is open

    cv2.imshow('video', frame)  #show the frame

    key = cv2.waitKey(1)        #key captures key from the key board
    if key == 27:               #27 -escape
        break

cam.release()                   #stop the video
cv2.destroyAllWindows()         #closes everything