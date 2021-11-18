import cv2 as cv
import os
from teachable import Model
dirname, filename = os.path.split(os.path.abspath(__file__))

def watchStream(source=0):
    model = Model(path="{}/keras_model.h5".format(dirname), pathLabel="{}/labels.txt".format(dirname))
    capture = cv.VideoCapture(source)

    while True:
        ret, frame = capture.read()
        if source == 0:
            frame = cv.flip(frame, 1)

        # In case the image is not read properly
        if not ret:
            continue

        results = model.predict(frame)
        #print results
        print(results)

        # Print the predicted label into the screen.
        cv.putText(frame, results['label'], (280, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        if cv.waitKey(1) and 0xff == ord('q'):
            exit = True
            break
        # Show the frame
        cv.imshow('Frame', frame)

    capture.release()
    cv.destroyAllWindows()

watchStream()