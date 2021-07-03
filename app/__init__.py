from flask import Flask
import app.CONFIG as CONFIG
import os


# create necessary paths

if not os.path.exists(CONFIG.DATASET_DIR_PATH):
    os.makedirs(CONFIG.DATASET_DIR_PATH)

if not os.path.exists(CONFIG.TEMP_FILES_PATH):
    os.makedirs(CONFIG.TEMP_FILES_PATH)


app = Flask(__name__)
app.secret_key = "hhfsdfhs00390dsafjsdafkh30940"
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

from app import views
