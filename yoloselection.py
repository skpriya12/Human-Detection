import cv2
import time
import datetime
from ultralytics import YOLO

# Load YOLOv8 Model
try:
    model = YOLO("yolov8s.pt")  # Ensure the model file exists
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit(1)

# Open Video File
video_path = "/Users/karthick/CV/HumanDetection/twoqueue.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}. Check the file path.")
    exit(1)

# Zone selection variables
zones = {"A Zone": None, "B Zone": None}
drawing = False
start_point = None


# Mouse callback function
def draw_zone(event, x, y, flags, param):
    global drawing, start_point, zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        if zones["A Zone"] is None:
            zones["A Zone"] = (start_point, end_point)
            print(f"A Zone set: {zones['A Zone']}")
        elif zones["B Zone"] is None:
            zones["B Zone"] = (start_point, end_point)
            print(f"B Zone set: {zones['B Zone']}")
        else:
            # Overwrite Zones (3rd zone replaces A, 4th zone replaces B)
            zones["A Zone"], zones["B Zone"] = zones["B Zone"], (start_point, end_point)
            print(f"A Zone updated: {zones['A Zone']}")
            print(f"B Zone updated: {zones['B Zone']}")


cv2.namedWindow("Define Zones")
cv2.setMouseCallback("Define Zones", draw_zone)

# Frame processing settings (every 10 seconds)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * 10  # Process once every 10 seconds
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame. Exiting...")
        break  # Exit if video ends or read fails

    # Show zone selection screen
    while zones["B Zone"] is None:  # Wait for user to select both zones
        temp_frame = frame.copy()
        if zones["A Zone"]:
            cv2.rectangle(temp_frame, zones["A Zone"][0], zones["A Zone"][1], (0, 255, 0), 2)
        if zones["B Zone"]:
            cv2.rectangle(temp_frame, zones["B Zone"][0], zones["B Zone"][1], (255, 0, 0), 2)

        cv2.imshow("Define Zones", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    frame_count += 1

    # Process every 10 seconds
    if frame_count % frame_interval == 0:
        results = model(frame)

        zone_counts = {"A Zone": 0, "B Zone": 0}

        # Check detected objects
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls[0])  # Get class index
                if cls == 0:  # YOLO class 0 = "person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    # Draw bounding boxes on detected humans
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box

                    # Count humans in each zone
                    for zone_name, zone in zones.items():
                        if zone:
                            (zone_x1, zone_y1), (zone_x2, zone_y2) = zone
                            if zone_x1 <= x_center <= zone_x2 and zone_y1 <= y_center <= zone_y2:
                                zone_counts[zone_name] += 1

        # Print results with timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - A Zone: {zone_counts.get('A Zone', 0)}, B Zone: {zone_counts.get('B Zone', 0)}")

    # Show live video with zones
    if zones["A Zone"]:
        cv2.rectangle(frame, zones["A Zone"][0], zones["A Zone"][1], (0, 255, 0), 2)
    if zones["B Zone"]:
        cv2.rectangle(frame, zones["B Zone"][0], zones["B Zone"][1], (255, 0, 0), 2)

    cv2.imshow("Video Feed", frame)

    # Exit Condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("User requested exit. Closing...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
