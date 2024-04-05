import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2


video_path = r"sample.mp4"
video_output = "speed_detection.mp4"

def get_speed(video_path, video_output):
    model = YOLO("best_model_Yv8_epoch41.pt")
    names = model.model.names

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(video_output,
                                cv2.VideoWriter_fourcc(*'mp4v'),  # Utiliser mp4v pour MP4
                                fps,
                                (w, h))
    line_pts = [(1000, 1000), (3500, 1000)]

    # Initialisation speed-estimation obj
    speed_obj = speed_estimation.SpeedEstimator()

    # Frame de départ
    frame_count = 0

    while cap.isOpened():
        speed_obj.set_args(reg_pts=line_pts,
                        names=names,
                        spdl_dist_thresh=20,
                        #    spdl_dist_thresh=500,
                        view_img=True)
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        if frame_count % 2 == 0:  # Ne traiter que les frames impaires
            tracks = model.track(im0, persist=True, show=False)
            im0 = speed_obj.estimate_speed(im0, tracks)
            video_writer.write(im0)

            if cv2.waitKey(1000) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()