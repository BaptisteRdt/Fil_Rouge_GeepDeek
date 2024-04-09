import cv2
from ultralytics import YOLO
from traffic_jam import traffic_jam

# Load the YOLOv8 model
model = YOLO('models/best_model_Yv8_epoch41.pt')

# Open the video file
video_path = ("data/croisement.mp4")

cap = cv2.VideoCapture(video_path)

# frame rate
frame_rate = 2
i = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        if i % frame_rate == 0:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            traffic_jam_bool = traffic_jam(results, 0.05)
            print(f'Is there any traffic jam on this frame? {traffic_jam_bool}')


            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
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