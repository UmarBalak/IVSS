import cv2
import math
# Load object detection model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
yolo_model = 'yolo_inference_graph.pb'
model = cv2.dnn_DetectionModel(yolo_model, config_file)
# tracker = cv2.TrackerMIL_create()
# Load class labels
classLabels = []
filename = 'Label.txt'
with open(filename, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    # print(classLabels)

# Set model parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Open video file
cap = cv2.VideoCapture("Thailand traffic footage for object recognition #4.mp4")
# cap = cv2.VideoCapture(0)

# Initialize variables for abnormal event detection
previous_boxes = None
previous_centroids = None
threshold_distance = 100
centre_point_pre = []
count = 0
car_count = 0
person_count = 0
tracking_object = {}
track_id = 0


# bbox = cv2.selectROI('Tracking', frame, False)
# tracker.init(frame, bbox)
# Loop over video frames
while True:
    ret, frame = cap.read()
    count +=1

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

    centre_point = []
    for box in bbox:
        (x, y, w, h) = box
        cy = int((y+y+h)/2)
        cx = int((x +x + w) / 2)
        centre_point.append((cx,cy))

        print("FRAME",count, " ",  x, y, w, h,)
        # cv2.rectangle( frame, (x,y), (x+w, y+h), (0,255,0), 2)
    if count <= 3:
        for pt in centre_point:
            for pt2 in centre_point_pre:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])

                if distance < 20:
                    tracking_object[track_id] = pt
                    track_id +=1

                    # if classLabels[classInd - 1] == 'person':
                    #     person_count += 1
    #
    else:
        for pt2 in centre_point:
            for object_id, pt2 in tracking_object.items():
                distance = math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])
                if distance <20:
                    tracking_object[object_id] = pt

    # for object_id, pt in tracking_object.items():
    for object_id, pt in tracking_object.items():
        cv2.circle(frame,pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1,(0,0,255), 2)

    if (len(classIndex) != 0):
        # Draw bounding boxes and labels for detected objects
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if (classInd <= 80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame,classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))
                if classLabels[classInd - 1] == 'car':
                    car_count += 1





        # Calculate centroids of detected objects
        centroids = [(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in bbox]


        # if previous_boxes is not None:
        #     if len(bbox) > len(previous_boxes):
        #         car_count+=(len(bbox) - len(previous_boxes))
        #
        # previous_boxes = bbox

        # Detect abnormal events based on changes in object movement
        if previous_centroids is not None:
            for i, centroid in enumerate(centroids):
                if i < len(previous_centroids):
                    distance = ((centroid[0] - previous_centroids[i][0]) ** 2 + (centroid[1] - previous_centroids[i][1]) ** 2) ** 0.5
                    # if distance > threshold_distance:
                    #     cv2.putText(frame, "Abnormal event detected", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255))
                        # Do something to handle the abnormal event here

        # Update previous boxes and centroids
        previous_boxes = bbox
        previous_centroids = centroids

    # Display current frame
    cv2.putText(frame, f'Car count: {car_count}', (8, 30),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))
    # cv2.putText(frame, f'Person count: {person_count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))
    cv2.imshow('object detection tutorial', frame)

    centre_point_pre = centre_point.copy()



    # Check for exit key
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllwindows()
