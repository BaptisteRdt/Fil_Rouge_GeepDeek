import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import csv
import re

video_path = r"Vidéo sans titre.mp4"
video_output = "output_video.mp4"
output_csv = "results.csv"

def process_video(video_path, output_video, output_csv):
    # Load the YOLOv8 model
    model = YOLO('license_plate_detector.pt')

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width // 2, frame_height // 2))

    # Set the desired frame rate
    frame_rate = 4

    # Initialize variables for previous license plate
    prev_license_plate = ""

    # Set for storing detected vehicle IDs
    detected_vehicles = set()

    # Regular expression to match alphanumeric characters
    pattern = re.compile('[a-zA-Z0-9]+')

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Frame', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'License Plate Text', 'Confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        i = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if i % frame_rate == 0:
                    # Run YOLOv8 detection on the frame
                    results = model.predict(frame, conf=0.1)

                    # Sort results based on confidence
                    results.sort(key=lambda x: x[0][2] if len(x[0]) > 2 else 0, reverse=True)

                    for item in results:
                        # Extract bounding box coordinates
                        bbox_data = item.boxes.data.cpu().numpy()[0]
                        xmin, ymin, xmax, ymax = map(int, bbox_data[:4])
                        
                        # Crop the license plate region
                        plate_img = frame[ymin:ymax, xmin:xmax]

                        # Apply pre-processing to the plate image
                        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        try:
                            # Use EasyOCR to recognize the text
                            result = reader.readtext(plate_thresh)

                            if result:
                                text = result[0][-2]
                                confidence = result[0][-1]

                                # Filter out spaces and special characters
                                filtered_text = ''.join(pattern.findall(text))

                                # Check confidence threshold and text length
                                if confidence > 0.3 and len(filtered_text) >= 7:
                                    # Check if the vehicle is already detected
                                    vehicle_id = (xmin, ymin)  # Using the top-left corner as the vehicle ID
                                    if vehicle_id not in detected_vehicles:
                                        detected_vehicles.add(vehicle_id)

                                        # Compare with previous license plate
                                        if filtered_text != prev_license_plate:
                                            prev_license_plate = filtered_text

                                            # Draw bounding box on the frame
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                                            # Display recognized text and confidence on the frame
                                            cv2.putText(frame, f"{filtered_text} ({confidence:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                                            
                                            # Write to CSV file if confidence > 0.1 and text length >= 7
                                            writer.writerow({
                                                'Frame': i,
                                                'Xmin': xmin,
                                                'Ymin': ymin,
                                                'Xmax': xmax,
                                                'Ymax': ymax,
                                                'License Plate Text': filtered_text,
                                                'Confidence': confidence
                                            })
                        except Exception as e:
                            print(f"An error occurred: {e}")

                    # Resize the frame
                    frame_resized = cv2.resize(frame, (frame_width // 2, frame_height // 2))

                    # Write the annotated frame to the output video
                    out.write(frame_resized)

                    # Display the annotated frame
                    cv2.imshow("YOLOv8 + EasyOCR Detection", frame_resized)

                    if cv2.waitKey(1000 // frame_rate) & 0xFF == ord("q"):
                        break
            else:
                break
            i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Appel de la fonction principale
if __name__ == "__main__":
    process_video(video_path, video_output, output_csv)
