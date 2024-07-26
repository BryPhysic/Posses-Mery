import os
import cv2
import pandas as pd


video_folder = 'Data/UBI_FIGHTS/videos'
annotation_folder = 'Data/UBI_FIGHTS/annotation'
output_folder = 'Images/datos/'


fight_folder = os.path.join(output_folder, 'fight')
no_fight_folder = os.path.join(output_folder, 'no_fight')

os.makedirs(fight_folder, exist_ok=True)
os.makedirs(no_fight_folder, exist_ok=True)

def extract_and_label_frames(video_path, csv_path, fight_folder, no_fight_folder, frame_interval=1):
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    labels = pd.read_csv(csv_path, header=None)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            label = labels.iloc[frame_count, 0]
            frame_name = f"{video_name}_frame_{frame_count}.jpg"
            
            if label == 1:
                frame_path = os.path.join(fight_folder, frame_name)
                
                label_file_path = frame_path.replace('.jpg', '.txt')
                with open(label_file_path, 'w') as f:
                   # aun debo saber que tipo de etiqueta voy a usar (roboflow)
                    f.write("0 0.5 0.5 1.0 1.0") # etiqueta generica para probar
                frame_path = os.path.join(no_fight_folder, frame_name)
            
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()


for subdir in ['fight', 'normal']:
    video_subfolder = os.path.join(video_folder, subdir)
    for video_file in os.listdir(video_subfolder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_subfolder, video_file)
            video_name = os.path.basename(video_path).split('.')[0]
            csv_path = os.path.join(annotation_folder, f"{video_name}.csv")
            
            if os.path.exists(csv_path):
                extract_and_label_frames(video_path, csv_path, fight_folder, no_fight_folder)
            else:
                print(f"No se encontró el archivo CSV para {video_name}")

print("Procesamiento completado.")