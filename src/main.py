from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import uvicorn
import shutil
import time
import cv2
import uuid
import torch
import threading
import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# Définir le dossier de téléchargement et s'assurer qu'il existe
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_PROCESSED_FOLDER = 'uploads_processed/'
if not os.path.exists(UPLOAD_PROCESSED_FOLDER):
    os.makedirs(UPLOAD_PROCESSED_FOLDER)

MODEL_FOLDER = 'model/'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

app = FastAPI()
templates = Jinja2Templates(directory="templates")



# Montage des dossiers statiques pour servir les fichiers CSS, JS, etc.
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads_processed", StaticFiles(directory="uploads_processed"), name="uploads_processed")

# Dictionnaire pour mapper les identifiants uniques des vidéos vers leur chemin et modèle associés
file_mappings = {}

# Variables globales pour stocker les résultats de détection et de vitesse
fps = 0.0
results = {"car": 0, "truck": 0}

# Verrouillage pour assurer l'accès sûr aux résultats partagés entre les threads
results_lock = threading.Lock()

# Variable globale pour contrôler l'arrêt du traitement vidéo
should_continue = True


# Page d'accueil de l'application, affichant un formulaire de téléversement de vidéo
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route pour téléverser une vidéo et spécifier le modèle à utiliser
@app.post("/upload/")
async def upload_file(file: UploadFile, model: str = Form(...)):
    # Sécurisation du nom de fichier pour éviter les attaques
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    #modelpath = 'model/' + model
    modelpath = os.path.join(MODEL_FOLDER, model)
    # Copie du fichier téléversé vers le dossier de téléchargement
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Génération d'un identifiant unique pour cette vidéo
    unique_id = str(uuid.uuid4())
    # Enregistrement du chemin de la vidéo et du chemin du modèle associé
    file_mappings[unique_id] = {'path': filepath, 'model': modelpath}
    # Redirection vers la page de visualisation de la vidéo téléversée
    return RedirectResponse(app.url_path_for('view_video', unique_id=unique_id))


# Route pour afficher la vidéo téléversée
@app.post("/view_video/{unique_id}", response_class=HTMLResponse)
async def view_video(request: Request, unique_id: str):
    return templates.TemplateResponse("view_video.html", {"request": request, "unique_id": unique_id})


# Route pour traiter la vidéo téléversée et renvoyer un flux d'images traitées
@app.get("/process_video/{unique_id}")
async def process_video_endpoint(unique_id: str):
    # Récupération du chemin de la vidéo et du modèle associé à partir de l'identifiant unique
    filename = file_mappings.get(unique_id)
    if not filename:
        raise HTTPException(status_code=404, detail="File not found")
    # Appel à la fonction de traitement de la vidéo et renvoi d'un flux d'images traitées
    return StreamingResponse(process_video(filename['path'], filename['model']),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# Route pour obtenir les comptages de voitures et de camions
@app.get("/get_counts")
async def get_counts():
    # Renvoie les résultats de détection sous forme de réponse JSON
    response = JSONResponse(content=results)
    # Désactivation de la mise en cache pour obtenir les résultats en temps réel
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Route pour obtenir le nombre d'images par seconde traitées
@app.get("/get_fps")
async def get_fps():
    # Renvoie le nombre d'images par seconde sous forme de réponse JSON
    response = JSONResponse(content=fps)
    return response


# Route pour arrêter le traitement de la vidéo et afficher les résultats finaux
@app.get("/video_processed")
async def stop_processing(request: Request):
    global should_continue
    should_continue = False
    return templates.TemplateResponse("results.html", {"request": request})


# Fonction asynchrone pour traiter la vidéo en utilisant le modèle YOLO
async def process_video(video_path, model_name):
    model = YOLO(model_name)
    global results, fps, should_continue
    should_continue = True
    # Ouverture de la vidéo pour traitement
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Création d'une vidéo de sortie pour enregistrer les images traitées
    output_video = cv2.VideoWriter(os.path.join(UPLOAD_PROCESSED_FOLDER, 'video_processed.mp4'),
                                   cv2.VideoWriter_fourcc(*'avc1'), 30, (frame_width, frame_height))
    # Points de référence pour l'estimation de la vitesse des véhicules
    line_pts = [(0, 500), (1280, 500)]
    # Initialisation de l'objet d'estimation de la vitesse
    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts,
                       names=model.model.names)
    try:
        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                break

            car_count, truck_count = 0, 0
            start_time = time.time()
            # Détection des véhicules dans la frame actuelle
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, stream_buffer=True)

            if result[0].boxes.cls is not None:
                classes_tensor = result[0].boxes.cls
                valeurs_uniques, occurrences = torch.unique(classes_tensor, return_counts=True)
                class_labels = {int(cls): label for cls, label in model.names.items()}

                for cls, occ in zip(valeurs_uniques, occurrences):
                    if class_labels[int(cls)] == "car":
                        car_count = int(occ)
                    elif class_labels[int(cls)] == "truck":
                        truck_count = int(occ)

            # Estimation de la vitesse des véhicules dans la frame
            frame = speed_obj.estimate_speed(frame, result)
            # Affichage des boîtes de détection et des informations de vitesse
            frame_f = result[0].plot(probs=False, kpt_line=False)
            # Écriture de la frame traitée dans la vidéo de sortie
            output_video.write(frame_f)

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
            # Mise à jour des résultats de détection
            results["car"] = car_count
            results["truck"] = truck_count

    finally:
        # Libération des ressources vidéo à la fin du traitement
        output_video.release()
        cap.release()


# Lancement de l'application FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)