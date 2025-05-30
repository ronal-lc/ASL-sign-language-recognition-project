import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import time
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'
data = []
labels = []

# Se espera que cada mano tenga 21 puntos clave, con coordenadas (x, y) => 21 * 2 = 42
EXPECTED_LANDMARK_LENGTH = 42

def process_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            x_min, y_min = coords.min(axis=0)
            coords[:, 0] -= x_min
            coords[:, 1] -= y_min
            landmarks.extend(coords.flatten())
        if len(landmarks) == EXPECTED_LANDMARK_LENGTH:
            return landmarks  # Solo devuelve si la longitud es la esperada
    return None

start_time = time.time()
total_images = sum([len(files) for r, d, files in os.walk(DATA_DIR) if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files)])

with tqdm(total=total_images, desc="Procesando imágenes", dynamic_ncols=True) as pbar:
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error al cargar la imagen: {img_path}")
                        continue

                    # Procesar la imagen original
                    landmarks = process_image(img)
                    if landmarks:
                        data.append(landmarks)
                        labels.append(label)

                    # Reflejar la imagen horizontalmente
                    img_flipped = cv2.flip(img, 1)
                    landmarks_flipped = process_image(img_flipped)
                    if landmarks_flipped:
                        data.append(landmarks_flipped)
                        labels.append(label)

                    pbar.update(1)

# Guardar el dataset solo si los datos están completos
if data and labels:
    with open('data_signs.pickle', 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
    print(f"\nProcesadas {len(data)} imágenes (incluyendo reflejadas).")
else:
    print("\nNo se procesaron datos. Verifique su directorio de datos y archivos de imágenes.")

end_time = time.time()
print(f"Tiempo total de procesamiento: {end_time - start_time:.2f} segundos.")
