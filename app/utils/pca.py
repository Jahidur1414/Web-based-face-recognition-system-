import numpy as np
import cv2
import scipy.linalg as s_linalg


class PCA:

    def __init__(self, images, y, target_names, no_of_elements, quality_percent):
        self.images = np.asarray(images)
        self.y = y
        self.target_names = target_names
        self.no_of_elements = no_of_elements

        self.mean_face = np.asmatrix(np.mean(self.images, axis=1)).T
        self.images = self.images - self.mean_face
        self.quality_percent = quality_percent

    def give_p(self, d):
        sum = np.sum(d)
        sum_85 = self.quality_percent * sum/100
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    def reduce_dim(self):
        p, d, _ = s_linalg.svd(self.images, full_matrices=True)
        p_matrix = np.matrix(p)
        p = self.give_p(d)
        self.new_bases = p_matrix[:, 0:p]
        self.new_coordinates = np.dot(self.new_bases.T, self.images)

    def original_data(self, new_coordinates):
        return self.mean_face + (np.dot(self.new_bases, new_coordinates.T))

    def new_cord(self, name, img_height, img_width):
        img = cv2.imread(name)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        img_vec = np.asmatrix(gray).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def new_cord_for_image(self, image):
        img_vec = np.asmatrix(image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec) / (len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def recognize_face(self, new_cord_pca, k=0, dist_threshold=100000):
        classes = len(self.no_of_elements)
        start = 0
        distances = []
        for i in range(classes):
            # extract training PCA images for specific class
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.no_of_elements[i])]
            mean_temp = np.mean(temp_imgs, axis=1)
            start = start + self.no_of_elements[i]
            dist = np.linalg.norm(new_cord_pca - mean_temp)
            distances.append(dist)
        min = np.argmin(distances)

        # print(f"minimum L2 dist: {distances[min]}")
        #Temp Threshold
        if distances[min] < dist_threshold:
            # print("Person", k, ":", min, self.target_names[min])
            return self.target_names[min]
        else:
            # print("Person", k, ":", min, 'Unknown')
            return 'Unknown'