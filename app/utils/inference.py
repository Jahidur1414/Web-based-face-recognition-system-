import os
import cv2
import numpy as np
import app.CONFIG as CONFIG
from imutils.video import WebcamVideoStream
import xlsxwriter
import csv
import pandas as pd
import datetime
import time


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)

# import classes
from app.utils.dataset import DatasetLoader
from app.utils.face_alignment import FaceAligner
from app.utils.utils import ImagesToMatrix
from app.utils.pca import PCA
from app.mtcnn import MTCNN

namelist = []


class Infernce(object):
    def __init__(self, resize_scale=1, image_size=224):
        self.datast_loader = DatasetLoader(CONFIG.DATASET_DIR_PATH)
        # dataset parameters
        self.images_names = self.datast_loader.images_name_for_train
        self.y = self.datast_loader.y_for_train
        self.no_of_elements = self.datast_loader.no_of_elements_for_train
        self.target_names = self.datast_loader.target_name_as_array
        self.stream = WebcamVideoStream(src=0).start()
        self.resize_scale = resize_scale
        self._classes = os.listdir(CONFIG.DATASET_DIR_PATH)

        self.detector = MTCNN()
        self.face_aligner = FaceAligner(desiredFaceWidth=image_size)

        self.no_registered_people = False
        if len(self.images_names) == 0:
            self.no_registered_people = True
        else:
            # training image size
            self.img_width, self.img_height = 64, 64
            i_t_m_c = ImagesToMatrix(self.images_names, self.img_width, self.img_height)
            scaled_face = i_t_m_c.get_matrix()
            self.pca = PCA(scaled_face, self.y, self.target_names, self.no_of_elements, 90)
            self.pca.reduce_dim() # perfomr PCA
        #self.name = ""


    def __del__(self):
        self.stream.stop()

    def check(self, name):
        print(name)
        print("Yeah! It's Work !")
    

    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    def savedata(self, name):
        print("Start")
        self.assure_path_exists("Attendance/")
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
        attendance = [str(name), '', str(date), '', str(timeStamp)]
        col_names = ['Name', '', 'Date', '', 'Time']
        if exists:
            with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(attendance)
            csvFile1.close()
        else:
            with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(col_names)
                writer.writerow(attendance)
            csvFile1.close()
        print("Recognized person data saved !")



    def inference(self):    
        frame_orig = self.stream.read()
        # resize the frame to increase the speed
        frame = frame_orig
        frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)

        if not self.no_registered_people:
            face_bboxes = self.detector.detect_faces(frame)
            for face in face_bboxes:
                x1, y1, width, height = face['box']
                x2, y2 = x1 + width, y1 + height

                y1, x1 = int(y1 / self.resize_scale), int(x1 / self.resize_scale)
                y2, x2 = int(y2 / self.resize_scale), int(x2 / self.resize_scale)

                left_eye, right_eye = face['keypoints']['left_eye'], face['keypoints']['right_eye']
                aligned_face = self.face_aligner.align(frame, right_eye, left_eye)
                roi_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

                scaled = cv2.resize(roi_gray, (self.img_height, self.img_width))
                new_cord = self.pca.new_cord_for_image(scaled)
                prediction = self.pca.recognize_face(new_cord, dist_threshold=2200)
                if prediction == "unknown":
                    # Crop the image frame into rectangle
                    cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_orig, prediction, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Crop the image frame into rectangle
                    cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_orig, prediction, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    #namelist.append(prediction)
                    #--------------------------------------------------( Let's Start the change )----------------------------------------------------
                    #print(len(namelist))
                    
                    if len(namelist)==0:
                        #print("Hei mama")
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        print(prediction)
                        print(date)
                        print(timeStamp)
                        namelist.append(prediction)
                        self.savedata(prediction)
                    for i in range(len(namelist)):
                        #print(len(namelist))
                        #print("It's Work !")
                        self.check(namelist[i])
                        if namelist[i] != prediction:
                            #print("Hei mama")
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            print(prediction)
                            print(date)
                            print(timeStamp)
                            namelist.append(prediction)
                            self.savedata(prediction)

                    #namelist.append(prediction)



        else:
            _str = "No face is registered to recognise"
            cv2.putText(frame_orig, _str, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame_orig)
        data = []
        data.append(jpeg.tobytes())
        return data


def inference_webcam(camera):
    while True:
        data = camera.inference()
        frame = data[0]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
