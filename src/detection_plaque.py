import cv2
import re
import easyocr
import time
import csv
from ultralytics import YOLO
import os

# Définir le dossier de téléchargement et s'assurer qu'il existe
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_PROCESSED_FOLDER = 'uploads_processed/'
if not os.path.exists(UPLOAD_PROCESSED_FOLDER):
    os.makedirs(UPLOAD_PROCESSED_FOLDER)

# Fonction asynchrone pour traiter la vidéo en utilisant le modèle YOLO
async def detection_plaque(video_path, model_name):
    output_csv = "results.csv"

    model = YOLO(model_name)
    #model = 'model/' + model_name
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
    output_video = cv2.VideoWriter(os.path.join(UPLOAD_PROCESSED_FOLDER, 'video_processed.mp4'), fourcc, fps, (frame_width // 2, frame_height // 2))

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
        try :
            start_time = time.time()
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
                                                cv2.putText(frame, f"{filtered_text} ({confidence:.2f})", (xmin, ymin - 10),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

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
                        output_video.write(frame_resized)

                # Encodage de l'image traitée pour le streaming
                _, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = buffer.tobytes()
                end_time = time.time()

                # Calcul du temps d'inférence et du nombre d'images par seconde
                inference_time = end_time - start_time
                if inference_time > 0:
                    fps = 1 / inference_time
                else:
                    fps = 0

                # Renvoi de l'image traitée dans le flux
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

        finally:
            # Libération des ressources vidéo à la fin du traitement
            output_video.release()
            cap.release()

