import cv2
from ultralytics import YOLO
from traffic_jam import traffic_jam
from vehicle_dictionary import get_vehicles_dict

# Load the YOLOv8 model
model = YOLO('models/best_model_Yv8_epoch41.pt')

# Open the video file
video_path = ("data/sample.mp4")

cap = cv2.VideoCapture(video_path)

# frame rate
frame_rate = 2
vehicles = None
i = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        if i % frame_rate == 0:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            vehicles = get_vehicles_dict(results, vehicles)
            traffic_jam_bool = traffic_jam(vehicles)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
            annotated_frame = cv2.putText(img=annotated_frame, org=(50, 50),
                                          color=(0, 0, 255) if traffic_jam_bool else (0, 255, 0), thickness=2,
                                          text="Traffic jam !!!" if traffic_jam_bool else "Fluid traffic",
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    i += 1

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()