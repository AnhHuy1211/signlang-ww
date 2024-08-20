import cv2
import numpy as np
from collections import defaultdict
import tensorflow as tf
import base64

tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def process_each_frame(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # Run inference
    detection = model(input_tensor)
    return detection


def run_detection(model, src):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    scores = []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        # Actual detection.
        detection = process_each_frame(model, image_np)
        if np.any(detection['detection_scores'] > 0.1):
            scores.append(np.max(detection["detection_scores"]))
            frames.append(frame)
    cap.release()
    return scores, frames


def categorize_scores(scores, thresholds):
    if len(scores) == 0:
        return [[-1, 0, 100]]
    categories = []
    for score in scores:
        if score >= thresholds[0]:
            category = 1  # Category for high scores
        elif score >= thresholds[1]:
            category = 0  # Category for medium scores
        else:
            category = -1  # Category for low scores
        categories.append([score, category])
    # Use defaultdict to create a dictionary with default value of 0
    grouped_data = defaultdict(int)
    # Iterate through the data and count occurrences of the second element
    for row in categories:
        grouped_data[row[1]] += 1
    # Total number of elements
    total_elements = len(scores)
    judgement = []
    # Print the grouped data with percentages
    for key, value in grouped_data.items():
        percentage = round((value / total_elements) * 100)
        judgement.append([key, value, percentage])

    return judgement


def get_final_judgment(judgment: list):
    max_row = max(judgment, key=lambda x: x[2])
    if max_row[0] == 1:
        return "high", max_row[2]
    elif max_row[0] == 0:
        return "potential", max_row[2]
    else:
        return "low", max_row[2]


def extract_frames(scores, frames):
    scores = np.array(scores)
    min_index = np.argmin(scores)
    min_frame = frames[min_index]
    _, buffer_min = cv2.imencode('.jpg', min_frame)
    base64_min = base64.b64encode(buffer_min).decode('utf-8')

    max_index = np.argmax(scores)
    max_frame = frames[max_index]
    _, buffer_max = cv2.imencode('.jpg', max_frame)
    base64_max = base64.b64encode(buffer_max).decode('utf-8')

    average = sum(scores) / len(scores)
    avg_idx = (np.abs(scores - average)).argmin()
    avg_frame = frames[avg_idx]
    _, buffer_avg = cv2.imencode('.jpg', avg_frame)
    base64_avg = base64.b64encode(buffer_avg).decode('utf-8')

    return "data:image/jpeg;base64," + base64_min, "data:image/jpeg;base64," + base64_avg, "data:image/jpeg;base64," + base64_max
