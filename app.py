from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#class_labels = ["Anger", "Fear", "Happy", "Sad", "Surprise"]
class_labels = ["Happy", "Sad", "Surprise"]
model = load_model('model.h5') #5 emotions
#model = load_model('model.h5') #only 3 emotions

img = cv2.imread('happy_test1.png')[:,:,1]
print(img.shape)
img = cv2.resize(img, (48, 48))
img = img.reshape(48, 48, 1)
print(img.shape)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = tf.image.resize(img, [48, 48, 1])

plt.imshow(img)
plt.show()
img = tf.expand_dims(img, axis=0)
#class_labels_3 = ["Happy", "Sad", "Surprise"]
pred_ind = np.argmax(model.predict(img))
pred = class_labels[pred_ind]
print('predictions = ', pred)
