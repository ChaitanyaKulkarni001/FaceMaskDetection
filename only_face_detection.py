# import the necessary packages
import cv2
import imutils
import numpy as np

def detect_faces(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize list of faces and their corresponding locations
    faces = []
    locs = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # add the face coordinates to the list
            locs.append((startX, startY, endX, endY))

    return locs

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream
    ret, frame = vs.read()
    if not ret:
        break

    # resize the frame for faster processing
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame
    face_locs = detect_faces(frame, faceNet)

    # loop over the detected face locations
    for (startX, startY, endX, endY) in face_locs:
        # draw a rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
