import os

################ Configuration File #######################

TEMP_FILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/temp')
DATASET_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/data/registered_people_data')

################# Parameters to Consider #####################

TRAINING_IMAGES = 3
RESIZE_SCALE = 1.0

##############################################################