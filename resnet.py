from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
import sys

model = ResNet50(weights="imagenet")

image = cv2.imread(sys.argv[1])

images = np.array([
    cv2.resize(image, (224, 224)),
    np.pad(cv2.resize(image, (112, 112)), [[56, 56], [56, 56], [0, 0]], "constant", constant_values=255),
    np.pad(cv2.resize(image, (112, 112)), [[0, 112], [0, 112], [0, 0]], "constant", constant_values=255)
])

predictions = model.predict(preprocess_input(images.astype(np.float32)))
print('Predicted:', decode_predictions(predictions, top=3))

for index, image in enumerate(images):

    cv2.imshow("image{}".format(index), image)

while cv2.waitKey(1000) != ord("q"):
    pass
