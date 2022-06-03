""""
Utsuk Paudayal
The British College
C7227233
"""

"""
CV2 is an open-cv module
Numpy is python library used for arrays
"""
import cv2
import numpy as np
import time

#Load Yolo (yolov3 weights and cfg for higher accuracy)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# yolo tiny for better fps but less accuracy
#net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

"""
#Loading Image
img = cv2.imread("./multiple.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
"""
cap = cv2.VideoCapture("./goal.mp4")

# VideoCapture(0) is for webcam
#cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id +=1

    #object detection through image
    #height, width, channels = img.shape

    height, width, channels = frame.shape

    #Detecting Objects from image
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    #this is used to produce 3 diffrent images for rgb
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Showing Informations on Screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object Detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

            #Rectangle Bounding Box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # number_objects_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # print(label) // this will show what objects are there in picture and label them in the console

            # Makes bounding box of rectangular shape for objects in image
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y +30), font, 1, color, 3)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 3)

    # for FPS count
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 4)

    # if want to dectect through image, change frame to img
    cv2.imshow("Utsuk's Object Detection", frame)

    # waitKey(o) helps to freeze the screen and focus on one image
    #cv2.waitKey(0)

    #  waitKey(1) makes screen to wait for 1ms and run the while loop after that.
    # Key helps to break the loop in real time when any key is pressed on keyboard
    key = cv2.waitKey(1)
    if key == 27:
        break

# cap release makes sure that camera is also turned off after we exit the program
cap.release()
cv2.destroyAllWindows()