import numpy as np
import cv2
import os
import shutil
from imutils.video import WebcamVideoStream

from app.mtcnn import MTCNN
from app.utils.face_alignment import FaceAligner
from app.utils.utils import _resize_and_pad

import app.CONFIG as CONFIG
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Capture_Images(object):
    def __init__(self, new_name, image_size=224):
        self.stream = WebcamVideoStream(src=0).start()
        self.registered = False
        self.name_to_register = new_name
        self.detector = MTCNN()
        self.face_aligner = FaceAligner(desiredFaceWidth=image_size)
        self.image_counter = 0

        # check if the person has not already been registered
        if os.path.exists(os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register)):
            # remove if there's unsuccessful registeration
            if len(os.listdir(os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register))) < CONFIG.TRAINING_IMAGES:
                shutil.rmtree(os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register))
                os.makedirs(os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register))
            else:
                self.name_to_register = None
        else:
            os.makedirs(os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register))

    def __del__(self):
        self.stream.stop()

    def capture_and_process_images(self):
        frame = self.stream.read()

        if self.name_to_register is None:
            _str = f"{self.name_to_register} is already Registered.\nTry Another name !!!"
            cv2.putText(frame, _str, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        elif self.registered:
            _str = f"{self.name_to_register} has been Registered !!! "
            cv2.putText(frame, _str, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        else:

            face_bboxes = self.detector.detect_faces(frame)
            # ignore the frame with no faces detected or multiple faces detected
            if len(face_bboxes) == 1:

                face = face_bboxes[0]
                x1, y1, width, height = face['box']
                x2, y2 = x1 + width, y1 + height

                left_eye, right_eye = face['keypoints']['left_eye'], face['keypoints']['right_eye']
                aligned_face = self.face_aligner.align(frame, right_eye, left_eye)
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

                # save face
                new_image_name = f"frame_{self.image_counter}__{self.name_to_register}.jpg"
                save_path = os.path.join(CONFIG.DATASET_DIR_PATH, self.name_to_register, new_image_name)
                cv2.imwrite(save_path, aligned_face)

                # Display the video frame, with bounded rectangle on the person's face, person name and frame count
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(frame, str(self.image_counter), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                _str = "Registering new person: " + self.name_to_register
                cv2.putText(frame, _str, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                self.image_counter += 1
            cv2.waitKey(50)

            # If image taken reach 20, stop taking video
            if self.image_counter >= CONFIG.TRAINING_IMAGES:
                self.registered = True

        ret, jpeg = cv2.imencode('.jpg', frame)
        data = []
        data.append(jpeg.tobytes())
        return data


def register_capture_images_(camera):
    while True:
        data = camera.capture_and_process_images()
        frame = data[0]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# this function takes images and process them
def process_existing_images(name):

    # check if there's any images to register
    if os.path.exists(CONFIG.TEMP_FILES_PATH):
        images = os.listdir(CONFIG.TEMP_FILES_PATH)
        if len(images) == 0:
            return {
                "status": False,
                "message": "No images found..."
            }
    else:
        return {
            "status": False,
            "message": "No images found..."
        }

    # check if the person has not already been registered
    if os.path.exists(os.path.join(CONFIG.DATASET_DIR_PATH, name)):
        # remove if there's unsuccessful registeration
        if len(os.listdir(os.path.join(CONFIG.DATASET_DIR_PATH, name))) < CONFIG.TRAINING_IMAGES:
            shutil.rmtree(os.path.join(CONFIG.DATASET_DIR_PATH, name))
            os.makedirs(os.path.join(CONFIG.DATASET_DIR_PATH, name))
        else:
            for image_path in os.listdir(CONFIG.TEMP_FILES_PATH):
                os.remove(os.path.join(CONFIG.TEMP_FILES_PATH, image_path))
            return {
                "status": False,
                "message": "current person has already been registered..."
            }
    else:
        os.makedirs(os.path.join(CONFIG.DATASET_DIR_PATH, name))

    is_registered = False
    detector = MTCNN()
    face_aligner = FaceAligner(desiredFaceWidth=224)
    image_counter = 0
    for image_name in os.listdir(CONFIG.TEMP_FILES_PATH):
        # if person already present in database

        # if the file is not image then proceed to the next file
        if image_name.split(".")[-1] not in ["jpg", "JPG", "png", "jpeg"]:
            continue
        
        image_path = os.path.join(CONFIG.TEMP_FILES_PATH, image_name)

        try:
            frame = cv2.imread(image_path)
            frame = _resize_and_pad(frame)

            # first detecting faces in an image (finding face locations)
            faces = detector.detect_faces(frame)
            
            # if an image has exactly one person                
            if len(faces) != 1:
                continue

            left_eye, right_eye = faces[0]['keypoints']['left_eye'], faces[0]['keypoints']['right_eye']
            aligned_face = face_aligner.align(frame, right_eye, left_eye)
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

            # save face
            _, extension = os.path.splitext(image_name)
            new_image_name = f"frame_{image_counter}__{extension}"
            save_path = os.path.join(CONFIG.DATASET_DIR_PATH, name, new_image_name)
            cv2.imwrite(save_path, aligned_face)
            image_counter += 1

            # if images has already been added, then save data, break the loop and proceed to the next person
            if image_counter >= CONFIG.TRAINING_IMAGES:
                is_registered = True
                break
        except Exception as e:
            pass
    
    # removing temp files
    for image_path in os.listdir(CONFIG.TEMP_FILES_PATH):
        os.remove(os.path.join(CONFIG.TEMP_FILES_PATH, image_path))

    if not is_registered:
        # removing extra files
        if os.path.exists(CONFIG.DATASET_DIR_PATH, name):
            shutil.rmtree(os.path.exists(CONFIG.DATASET_DIR_PATH, name))

        print("\nperson was not registered sucessfully due to the following reasons...")
        print("1. either the images contained multiple faces or no faces")
        print("2. the number of images were less than required number ({} default)\n".format(training_images))
        return {
            "status": False,
            "message": "Not registered due to wrong or less number of images"
        }

    return {
            "status": True,
            "message": f"{name} registered successfully !!!"
        }
