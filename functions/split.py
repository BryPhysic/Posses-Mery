import os
import cv2
import pandas as pd
import random
import shutil

# Definir las rutas
video_folder = 'Data/UBI_FIGHTS/videos'
annotation_folder = 'Data/UBI_FIGHTS/annotation'
output_folder = 'Images/datos/'


fight_folder = os.path.join(output_folder, 'fight')
no_fight_folder = os.path.join(output_folder, 'no_fight')

os.makedirs(fight_folder, exist_ok=True)
os.makedirs(no_fight_folder, exist_ok=True)

def extract_and_label_frames(video_path, csv_path, fight_folder, no_fight_folder, frame_interval=5):
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    labels = pd.read_csv(csv_path, header=None)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar solo cada "frame_interval" fotogramas
        if frame_count % frame_interval == 0:
            label = labels.iloc[frame_count, 0]
            frame_name = f"{video_name}_frame_{frame_count}.jpg"
            
            if label == 1:
                frame_path = os.path.join(fight_folder, frame_name)
                # aun debo guardar el archivo de texto con las coordenadas eso lo hcemos con roboflow
                label_file_path = frame_path.replace('.jpg', '.txt')
                #with open(label_file_path, 'w') as f:
                    
                    #f.write("0 0.5 0.5 1.0 1.0")
            else:
                frame_path = os.path.join(no_fight_folder, frame_name)
            
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

# Procesar todos los videos en la carpeta
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

# Función para dividir datos en entrenamiento y prueba
def split_data(source_folder, train_folder, test_folder, split_ratio=0.8):
    # Crear carpetas de entrenamiento y prueba
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Obtener todos los archivos en la carpeta origen
    files = os.listdir(source_folder)
    random.shuffle(files)  # Mezclar archivos aleatoriamente

    # Calcular el índice de división
    split_index = int(len(files) * split_ratio)

    # Mover archivos a las carpetas correspondientes
    for i, file in enumerate(files):
        src_path = os.path.join(source_folder, file)
        if i < split_index:
            dest_path = os.path.join(train_folder, file)
        else:
            dest_path = os.path.join(test_folder, file)
        
        shutil.move(src_path, dest_path)

# Rutas para carpetas de entrenamiento y prueba
train_fight_folder = os.path.join(output_folder, 'train/fight')
test_fight_folder = os.path.join(output_folder, 'test/fight')
train_no_fight_folder = os.path.join(output_folder, 'train/no_fight')
test_no_fight_folder = os.path.join(output_folder, 'test/no_fight')

# Dividir datos de 'fight' y 'no_fight'
split_data(fight_folder, train_fight_folder, test_fight_folder, split_ratio=0.8)
split_data(no_fight_folder, train_no_fight_folder, test_no_fight_folder, split_ratio=0.8)

print("Datos divididos en entrenamiento y prueba.")