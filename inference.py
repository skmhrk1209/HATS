from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import cv2
import sys

model = InceptionV3(weights="imagenet")

for filename in sys.argv[1:]:

    image = cv2.imread(filename)

    images = np.array([
        np.pad(cv2.resize(image, (112, 112)), [[56, 56], [56, 56], [0, 0]], "constant", constant_values=255),
        np.pad(cv2.resize(image, (112, 112)), [[0, 112], [0, 112], [0, 0]], "constant", constant_values=255)
    ])

    predictions = model.predict(preprocess_input(images.astype(np.float32)))
    print("predicted:", decode_predictions(predictions, top=1))

    while True:

        if cv2.waitKey(1000) == ord("q"):
            break
