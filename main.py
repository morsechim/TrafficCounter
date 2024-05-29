import os
import time
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Define config
video_path = os.path.join(".", "videos", "traffic.mp4")
output_path = os.path.join(".", "videos", "output.mp4")
model_path = os.path.join(".", "weights", "yolov9c.pt")
conf_threshold = 0.5
tracking_class = 2 # None: track all

# mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
# define window name
cv2.namedWindow("Frame")
# set the mouse callback function
cv2.setMouseCallback("Frame", show_coordinates)

# Define DeepSORT instance
tracker = DeepSort(max_age=30)

# define limits line
limits_up = [459, 265, 657, 280]
limits_down = [684, 288, 885, 282]
# define total count list
total_count_up = []
total_count_down = []

# Define processing device mode
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(model_path).to(device)

# Load file classes.names
class_names_path = os.path.join(".", "data", "classes.names")
with open(class_names_path) as f:
    class_names = f.read().strip().split('\n')

# Random color each class name
colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(999)]
# colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

# Define video input instance
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video output instance
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# Define time to calculate FPS
previous_time = time.time()


# define ROI mask
mask_path = os.path.join(".", "images", "mask.png")
mask = cv2.imread(mask_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    # print(f"FPS: {fps:.2f}")
    
    # define detection roi
    detection_roi = cv2.bitwise_and(frame, mask)

    # Get detection results
    results = model(detection_roi)[0]

    detections = []
    for bbox in results.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        class_id = int(bbox.cls[0])
        confidence = float(bbox.conf[0])
        
        if (tracking_class is None and confidence >= conf_threshold) or (class_id == tracking_class and confidence >= conf_threshold):
            w, h = x2 - x1, y2 - y1
            detections.append([[x1, y1, w, h], confidence, class_id])

    # Update tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # counter up
    cv2.line(frame, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 4)
        
    # counter down
    cv2.line(frame, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 0, 255), 4)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            
            color = tuple(map(int, colors[class_id]))
            draw_color = (colors[int(track_id) % len(colors)])
            
            # color = colors[class_id]
            # B, G, R = map(int,color)
            # draw_color = (B, G, R)
            
            # calculate center object
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            # counter tracker up
            if limits_up[0] < cx < limits_up[2] and limits_up[1] - 5 < cy < limits_up[3] + 5:
                # check id not exites 
                if total_count_up.count(track_id) == 0:
                    total_count_up.append(track_id)
                    # holding line
                    cv2.line(frame, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)
            
            # counter tracker down
            if limits_down[0] < cx < limits_down[2] and limits_down[1] - 5 < cy < limits_down[3] + 5:
                # check id not exites 
                if total_count_down.count(track_id) == 0:
                    total_count_down.append(track_id)
                    # holding line
                    cv2.line(frame, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 5)
            
            # bbox label
            label = f"{class_names[class_id]} #{track_id}"
            
            # draw object bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            # draw center object 
            cv2.circle(frame, (cx, cy), 3, draw_color, cv2.FILLED)
            
            # calculate text size for dynamically words
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x2, y2 = x1 + text_width, y1 - text_height - baseline
            # draw background label
            cv2.rectangle(frame, (max(0, x1), max(35, y1)), (x2, y2), draw_color, cv2.FILLED)
            # draw confident score and class name
            cv2.putText(frame, label, (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # draw counter text
    cv2.putText(frame, f"OUT: {len(total_count_up)}", (frame.shape[1] - 300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"IN: {len(total_count_down)}", (frame.shape[1] - 300, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, lineType=cv2.LINE_AA)
    
    cv2.imshow("Frame", frame)
    cap_out.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()
