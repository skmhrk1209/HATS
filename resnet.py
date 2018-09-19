from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
import sys
'''
model = ResNet50(weights="imagenet")
image = cv2.imread(sys.argv[1])
image = cv2.resize(image, (112, 112))
image = np.pad(image, [[0, 112], [0, 112], [0, 0]], "constant")

cv2.imshow("image", image)
cv2.waitKey(1)

image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

predictions = model.predict(image)
print('Predicted:', decode_predictions(predictions, top=3)[0])
'''
model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print(img.dtype)
x = image.img_to_array(img)
print(x.dtype)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
