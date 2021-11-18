import cv2 as cv

def resizeFrame(frame,  scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def snapshots(source=0, output="/tmp", number=1):
    camera = cv.VideoCapture(source)

    i = 0
    while i < number:
        return_value, image = camera.read()
        cv.imwrite('samples/test/capture_' + str(i) + '.png', resizeFrame(image))
        i += 1

    del (camera)