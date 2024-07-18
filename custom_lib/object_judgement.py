import cv2
import numpy as np
from collections import defaultdict
from custom_lib.direct_handler import make_dir, join_path, get_file_list
import tensorflow as tf
from Tensorflow.models.research.object_detection.utils import config_util
from Tensorflow.models.research.object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.builders import model_builder
from custom_lib import config_tf as cfg


def load_model_from_checkpoint(checkpoint_index='ckpt-11'):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(cfg.CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(join_path([cfg.CHECKPOINT_PATH, 'ckpt-11'])).expect_partial()

    config = config_util.get_configs_from_pipeline_file(cfg.CONFIG_PATH)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(cfg.CONFIG_PATH, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = 5
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = join_path(
        [cfg.PRETRAINED_MODEL_PATH, "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8", "checkpoint", "ckpt-0"])
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = cfg.LABEL_MAP_PATH
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        join_path([cfg.TF_ANNOTATION_PATH, "train.record"])]
    pipeline_config.eval_input_reader[0].label_map_path = cfg.LABEL_MAP_PATH
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        join_path([cfg.TF_ANNOTATION_PATH, "test.record"])]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(cfg.CONFIG_PATH, "wb") as f:
        f.write(config_text)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(cfg.CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(join_path([cfg.CHECKPOINT_PATH, checkpoint_index])).expect_partial()
    return detection_model


@tf.function
def detect_fn(model, image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections


def run_inference(model, src):
    cap = cv2.VideoCapture(src)
    scores = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(model, input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        if np.any(detections['detection_scores'] > 0.1):
            scores.append(np.max(detections["detection_scores"]))
        image_np_with_detections = image_np.copy()
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    cap.release()
    return scores



def categorize_scores(scores, thresholds):
    """
    Categorizes scores in an array based on predefined thresholds.

    Args:
        scores: A list or NumPy array containing detection scores (0-1).
        thresholds: A list of thresholds (0-1) for different categories.

    Returns:
        A list of categories (strings) corresponding to each score in the input array.
    """
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
        return  "high"
    elif max_row[0] == 0:
        return "potential"
    else:
        return "low"
