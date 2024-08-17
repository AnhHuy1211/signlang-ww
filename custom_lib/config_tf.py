from custom_lib.direct_handler import make_dir, join_path, get_file_list

# TODO:  Set the Fully Qualified Path to the Template Repo Root Directory
TEMPLATE_ROOT_DIR = join_path(["Tensorflow"])

# You should not need to change these if you kept the directory structure from the Template repo
WORKSPACE_PATH = join_path([TEMPLATE_ROOT_DIR, "workspace"])
SCRIPTS_PATH = join_path([TEMPLATE_ROOT_DIR, "scripts"])
TF_MODEL_REPO_PATH = join_path([TEMPLATE_ROOT_DIR, "models"])
TF_ANNOTATION_PATH = WORKSPACE_PATH + '/tf-annotations'
IMAGES_PATH = WORKSPACE_PATH + '/images'
COLLECTED_IMAGES = join_path([IMAGES_PATH, "collected_images"])
TRAIN_IMAGES = join_path([IMAGES_PATH, "train"])
TEST_IMAGES = join_path([IMAGES_PATH, "test"])
MODEL_PATH = join_path([WORKSPACE_PATH, "models"])
PRETRAINED_MODEL_PATH = join_path([WORKSPACE_PATH, "pre-trained-models"])
CUSTOM_MODEL_DIR_NAME = 'my_ssd_mobnet'
EXPORTED_MODEL_PATH = make_dir(join_path([MODEL_PATH, CUSTOM_MODEL_DIR_NAME, "export", "saved_model", ""]))
CHECKPOINT_PATH = join_path([MODEL_PATH, "checkpoint"])
CONFIG_PATH = join_path([MODEL_PATH, CUSTOM_MODEL_DIR_NAME, "pipeline.config"])

# name of the label_map file that is created from the object class names
label_map_fname = "label_map.pbtxt"
LABEL_MAP_PATH = join_path([TF_ANNOTATION_PATH, label_map_fname])

##change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz'
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz'
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz'
    },
    'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz'
    },
    'ssd_mobilenet': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'pipeline.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    }
}

# TODO Set these hyperparameters as necessary
batch_size = 4
# The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing.
num_training_steps = 20000
# Perform evaluation after so many steps
num_eval_steps = 500
chosen_model = 'ssd_mobilenet'
model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

# TODO set the classes to be detected
classes = [
    "A",
    "B",
    "C",
    "D",
]
