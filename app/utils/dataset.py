
import os


class DatasetLoader:

    def __init__(self, dataset_path):

        # all images names e.g. [image1.jpg, image2.jpg .... imagen.jpg]
        self.images_name_for_train = []

        # names of persons (classes) e.g. ["a", "b", ... "n"]
        self.target_name_as_array = []

        # labels for persons e.g. [0, 1, 2, 3, 4, ... n]
        self.target_name_as_set = {}

        # all labels e.g. [0, 0, ..., 1, 1, 1, ..., n, n, n]
        self.y_for_train = []

        # number of images per person e.g. [10, 8, 29, 10, ... n]
        self.no_of_elements_for_train = []

        per_no = 0
        for name in os.listdir(dataset_path):

            dir_path = os.path.join(dataset_path, name)
            if not os.path.isdir(dir_path):
                continue

            for i, img_name in enumerate(os.listdir(dir_path)):
                img_path = os.path.join(dir_path, img_name)

                self.images_name_for_train.append(img_path)
                self.y_for_train.append(per_no)

                if len(self.no_of_elements_for_train) > per_no:
                    self.no_of_elements_for_train[per_no] += 1
                else:
                    self.no_of_elements_for_train.append(1)

                if i == 0:
                    self.target_name_as_array.append(name)
                    self.target_name_as_set[per_no] = name
            per_no += 1
        # print(self.target_name_as_array, self.target_name_as_set)
