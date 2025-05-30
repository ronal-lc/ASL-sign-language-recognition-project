import os
import sys
import cv2

DATA_DIR = './data'
NUMBER_OF_CLASSES = 27  # 26 letras más la clase "STOP"
DATASET_SIZE = 200
CAMERA_INDICES = [0, 1, 2]
WINDOW_NAME = 'Recolector de Imagenes'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def select_camera(indices):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Cámara abierta exitosamente con el índice {idx}")
            return cap
        cap.release()
    print("No se pudo abrir ninguna cámara. Verifique su conexión.")
    sys.exit(1)


def setup_window(window_name):
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def capture_initial_ready(cap, window_name):
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error al capturar el fotograma. Reintentando...")
            continue

        cv2.putText(frame, 'Listo? Presiona "Q" para comenzar', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(25) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        if key == ord('q'):  # Comienza la captura si presiona 'q'
            break
    return True


def capture_images(cap, class_dir, dataset_size, window_name):
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error al capturar el fotograma {counter}. Reintentando...")
            continue

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
        print(f"\rProgreso: {counter}/{dataset_size}", end="", flush=True)
    print()
    return True


def run_image_collection():
    ensure_dir(DATA_DIR)
    cap = select_camera(CAMERA_INDICES)
    setup_window(WINDOW_NAME)

    start_class = input("¿Deseas comenzar la recolección desde la clase 0? (s/n): ").strip().lower()

    if start_class == 'n':
        while True:
            try:
                start_class = int(input(f"Ingrese el número de la clase desde la que desea comenzar (0-{NUMBER_OF_CLASSES - 1}): "))
                if 0 <= start_class < NUMBER_OF_CLASSES:
                    break
                else:
                    print(f"Por favor, ingrese un número entre 0 y {NUMBER_OF_CLASSES - 1}.")
            except ValueError:
                print("Por favor, ingrese un número válido.")
    else:
        start_class = 0

    try:
        for cls in range(start_class, NUMBER_OF_CLASSES):
            if cls < 26:  # Clases de la A a la Z
                class_name = chr(65 + cls)  # Convierte el índice a letra
            else:  # Clase adicional "STOP"
                class_name = "STOP"
            class_dir = os.path.join(DATA_DIR, class_name)
            ensure_dir(class_dir)
            print(f'\nRecolectando datos para la clase {class_name}')

            if not capture_initial_ready(cap, WINDOW_NAME):
                print("\nRecolección de datos interrumpida.")
                break

            if not capture_images(cap, class_dir, DATASET_SIZE, WINDOW_NAME):
                print("\nRecolección de datos interrumpida.")
                break

            print(f"Clase {class_name} completada.")
        else:
            print("\nRecolección de datos finalizada.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


run_image_collection()
