import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

# =============================
# PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent  # carpeta app/

# Modelo de EMOCIONES (FER+)
EMOTION_MODEL_PATH = (BASE_DIR / "emotion-ferplus-8.onnx").resolve()

# Modelo YOLO de CARAS
YOLO_MODEL_PATH = (BASE_DIR / "yolov8n-face-lindevs.pt").resolve()

emotion_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}

print("[INFO] Emotion model path:", EMOTION_MODEL_PATH)
print("[INFO] YOLO model path:", YOLO_MODEL_PATH)

if not EMOTION_MODEL_PATH.exists():
    raise FileNotFoundError(f"Emotion ONNX no encontrado: {EMOTION_MODEL_PATH}")
if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO .pt no encontrado: {YOLO_MODEL_PATH}")

# =============================
# LOAD MODELS (CPU)
# =============================
print("[INFO] Loading YOLO face detector (CPU)...")
yolo = YOLO(str(YOLO_MODEL_PATH))

print("[INFO] Loading emotion model with ONNX Runtime (CPU)...")
sess = ort.InferenceSession(str(EMOTION_MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape  # puede contener None

# Detectar si el modelo espera NCHW o NHWC y cuántos canales
# Esperado típico FER+: (1,1,64,64) o (1,3,64,64)
if len(input_shape) != 4:
    raise RuntimeError(f"Modelo de emociones con input shape inesperado: {input_shape}")

def infer_emotion(face_bgr: np.ndarray) -> str:
    """
    face_bgr: imagen BGR recortada de la cara
    devuelve etiqueta de emoción
    """
    # Resize a 64x64
    face = cv2.resize(face_bgr, (64, 64))

    # Normalización típica FER+ (0..1 aproximadamente)
    # Many FER+ pipelines use: (x - 127) / 128
    face = face.astype(np.float32)
    face = (face - 127.0) / 128.0

    # Preparar tensor según shape esperado
    # Si input_shape[1] es 1 o 3 asumimos NCHW
    # Si input_shape[3] es 1 o 3 asumimos NHWC
    ch_nchw = input_shape[1] if isinstance(input_shape[1], int) else None
    ch_nhwc = input_shape[3] if isinstance(input_shape[3], int) else None

    if ch_nchw in (1, 3):  # NCHW
        if ch_nchw == 1:
            face_g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            face_g = (face_g - 127.0) / 128.0
            x = face_g[None, None, :, :]  # (1,1,64,64)
        else:
            # BGR -> RGB suele ser lo esperado en muchos ONNX
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            face_rgb = (face_rgb - 127.0) / 128.0
            x = np.transpose(face_rgb, (2, 0, 1))[None, :, :, :]  # (1,3,64,64)

    elif ch_nhwc in (1, 3):  # NHWC
        if ch_nhwc == 1:
            face_g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            face_g = (face_g - 127.0) / 128.0
            x = face_g[None, :, :, None]  # (1,64,64,1)
        else:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            face_rgb = (face_rgb - 127.0) / 128.0
            x = face_rgb[None, :, :, :]  # (1,64,64,3)

    else:
        # Si aquí cae, el modelo tiene un shape raro (ej: 224x224 o canales distintos)
        raise RuntimeError(f"No puedo inferir canales/layout desde input_shape: {input_shape}")

    # Inferencia ONNXRuntime (CPU)
    out = sess.run(None, {input_name: x})
    logits = out[0][0]  # (7,)

    pred_id = int(np.argmax(logits))
    return emotion_dict.get(pred_id, str(pred_id))


# =============================
# MAIN
# =============================
def FER_live_cam():
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = cv2.VideoWriter(
        "infer-yolo.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (frame_width, frame_height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # YOLO detección de caras (CPU)
        results = yolo(frame, conf=0.4, verbose=False)

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())

                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Emoción (CPU)
                try:
                    emotion = infer_emotion(face)
                except Exception as e:
                    emotion = "err"
                    # si quieres ver el error puntual:
                    # print("Emotion error:", e)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame, emotion, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        fps = 1.0 / (time.time() - start_time + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_video.write(frame)
        cv2.imshow("YOLO + Emotion (CPU)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER_live_cam()
