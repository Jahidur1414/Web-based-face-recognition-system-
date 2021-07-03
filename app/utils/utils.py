import cv2
import numpy as np
import app.CONFIG as CONFIG
import os


class ImagesToMatrix:

    def __init__(self, images_name, img_width, img_height):
        self.images_name = images_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = (img_width * img_height)

    def get_matrix(self):

        col = len(self.images_name)
        img_mat = np.zeros((self.img_size, col))

        i = 0
        for name in self.images_name:
            gray = cv2.imread(name, 0)
            gray = cv2.resize(gray, (self.img_height, self.img_width))
            mat = np.asmatrix(gray)
            img_mat[:, i] = mat.ravel()
            i += 1
        return img_mat


def _resize_and_pad(image, target_size=720):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def show_dataset():
    if os.path.exists(CONFIG.DATASET_DIR_PATH):
        people_registered = os.listdir(CONFIG.DATASET_DIR_PATH)

        return {
            "status": True,
            "Number of People Registered": len(people_registered),
            "People Registered": people_registered
        }
    else:
        return {
            "status": False
        }