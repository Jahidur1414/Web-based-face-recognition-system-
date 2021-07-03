import app.CONFIG as CONFIG
import os
import shutil


def delete_person(person_to_delete):
    if os.path.exists(os.path.join(CONFIG.DATASET_DIR_PATH, person_to_delete)):
        shutil.rmtree(os.path.join(CONFIG.DATASET_DIR_PATH, person_to_delete))
        return {
            "status": True,
            "message": f"{person_to_delete} has successfully deleted..."
        }
    else:
        return {
                "status": False,
                "message": f"The {person_to_delete} doesn't exists in the dataset..."
            }

