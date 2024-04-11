import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter, speed_estimation
from traffic_jam import traffic_jam
from vehicle_dictionary import get_vehicles_dict


async def model_final(video_path, model_name):
    # Load the YOLOv8 model
    model_vehicles = YOLO(model_name)
    names = model_vehicles.model.names

    # Open the video file
    video_path = video_path

    cap = cv2.VideoCapture(video_path)
    vehicles = None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output_video.mp4', fourcc, fps * 4,
    #                       (frame_width // 2, frame_height // 2))  # fps*4 pour accélérer la vidéo

    # Initialize ObjectCounter
    counter_object = object_counter.ObjectCounter()

    # Configurez les paramètres de ObjectCounter
    reg_pts = [(1000, 900), (3000, 800)]  # Exemple de points définissant une région
    counter_object.set_args(names, reg_pts)

    # Configurer la ligne permettant de détecter la vitesse d'un véhicule
    line_pts = [(1000, 1000), (3500, 1000)]

    # Initialisation speed-estimation obj
    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts, names=names, spdl_dist_thresh=20, view_img=True)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            vehicles_results = model_vehicles.track(frame, persist=True)

            vehicles = get_vehicles_dict(vehicles_results, vehicles)
            traffic_jam_bool = traffic_jam(vehicles)

            # Visualize the results on the frame
            annotated_frame = vehicles_results[0].plot()
            annotated_frame = speed_obj.estimate_speed(annotated_frame, vehicles_results)
            annotated_frame = counter_object.start_counting(annotated_frame, vehicles_results)
            annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
            annotated_frame = cv2.putText(img=annotated_frame, org=(50, 50),
                                          color=(0, 0, 255) if traffic_jam_bool else (0, 255, 0), thickness=2,
                                          text="Traffic jam !!!" if traffic_jam_bool else "Fluid traffic",
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

            # out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
