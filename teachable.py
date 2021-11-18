from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

class Model:
    def __init__(self, path="keras_model.h5", pathLabel="labels.txt"):
        self.model = load_model(path)
        self.labels = self.genLabels(pathLabel)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def genLabels(self, path="labels.txt"):
        labels = {}
        with open(path, "r") as label:
            text = label.read()
            lines = text.split("\n")
            for line in lines[0:-1]:
                hold = line.split(" ", 1)
                labels[hold[0]] = hold[1]
        return labels

    def predict(self, frame):
        frame = cv.resize(frame, (224, 224))
        image_array = np.asarray(frame)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        self.data[0] = normalized_image_array
        pred = self.model.predict(self.data)
        result = np.argmax(pred[0])
        label = self.labels[str(result)]

        return dict(result=result, label=label, pred=pred)