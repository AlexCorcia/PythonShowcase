import cv2
import numpy as np
import time
from pathlib import Path
from math import ceil
from cv2 import dnn

# =====================
# CONSTANTS
# =====================
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
threshold = 0.5

min_boxes = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0]
]

strides = [8.0, 16.0, 32.0, 64.0]

# =====================
# UTILS
# =====================
def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []

    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for _ in image_size:
        shrinkage_list.append(strides)

    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []

    for index in range(len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]

        for j in range(feature_map_list[1][index]):
            for i in range(feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    priors.append([
                        x_center,
                        y_center,
                        min_box / image_size[0],
                        min_box / image_size[1]
                    ])

    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]

    picked = []
    indexes = np.argsort(scores)

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked):
            break

        current_box = boxes[current]
        indexes = indexes[:-1]

        rest_boxes = boxes[indexes]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)

    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(width, height, confidences, boxes):
    boxes = boxes[0]
    confidences = confidences[0]

    picked_box_probs = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > threshold

        if not np.any(mask):
            continue

        subset_boxes = boxes[mask]
        box_probs = np.hstack([subset_boxes, probs[mask][:, None]])
        box_probs = hard_nms(box_probs, iou_threshold)

        picked_box_probs.append(box_probs)

    if not picked_box_probs:
        return np.empty((0, 4), dtype=int)

    picked = np.vstack(picked_box_probs)
    picked[:, [0, 2]] *= width
    picked[:, [1, 3]] *= height

    return picked[:, :4].astype(int)


def convert_locations_to_boxes(locations, priors):
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=2)


def center_to_corner(locations):
    return np.concatenate([
        locations[..., :2] - locations[..., 2:] / 2,
        locations[..., :2] + locations[..., 2:] / 2
    ], axis=2)

# =====================
# MAIN
# =====================
def FER_live_cam():
    BASE_DIR = Path(__file__).resolve().parent

    EMOTION_MODEL = BASE_DIR / "emotion-ferplus-8.onnx"
    FACE_MODEL = BASE_DIR / "RFB-320" / "RFB-320.caffemodel"
    FACE_PROTO = BASE_DIR / "RFB-320" / "RFB-320.prototxt"

    for p in [EMOTION_MODEL, FACE_MODEL, FACE_PROTO]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    emotion_net = cv2.dnn.readNetFromONNX(str(EMOTION_MODEL))
    face_net = dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))

    cap = cv2.VideoCapture(0)

    input_size = (320, 240)
    priors = define_img_size(input_size)

    emotions = [
        "neutral", "happiness", "surprise",
        "sadness", "anger", "disgust", "fear"
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        resized = cv2.resize(frame, input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        face_net.setInput(
            dnn.blobFromImage(rgb, 1 / image_std, input_size, 127)
        )

        boxes, scores = face_net.forward(["boxes", "scores"])

        boxes = convert_locations_to_boxes(
            boxes.reshape(1, -1, 4),
            priors
        )
        boxes = center_to_corner(boxes)
        boxes = predict(frame.shape[1], frame.shape[0], scores.reshape(1, -1, 2), boxes)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for x1, y1, x2, y2 in boxes:
            face = gray[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (64, 64))
            face = face.reshape(1, 1, 64, 64)

            emotion_net.setInput(face)
            pred = emotion_net.forward()[0]
            label = emotions[np.argmax(pred)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 5, 247), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (215, 5, 247), 2)

        fps = 1 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FER", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER_live_cam()
