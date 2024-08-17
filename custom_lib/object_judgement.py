import cv2
import numpy as np
from collections import defaultdict
from custom_lib.direct_handler import make_dir, join_path, get_file_list
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.builders import model_builder
from custom_lib import config_tf as cfg

tf.gfile = tf.io.gfile

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def run_inference_single_frame(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def run_inference(model, src):
    cap = cv2.VideoCapture(src)
    scores = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        # Actual detection.
        output_dict = run_inference_single_frame(model, image_np)
        if np.any(output_dict['detection_scores'] > 0.1):
            scores.append(np.max(output_dict["detection_scores"]))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return scores

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
        return "high"
    elif max_row[0] == 0:
        return "potential"
    else:
        return "low"
