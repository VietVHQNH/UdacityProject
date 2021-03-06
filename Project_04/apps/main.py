import cv2
import numpy as np

from tqdm import tqdm
from glob import glob

from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input, decode_predictions

from extract_bottleneck_features import *


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


class Engine:

    def __init__(self):
        self.main_model = Sequential()
        self.dog_names = open("data/dog_names.txt",'r', encoding = 'utf8').read().split("\n")
        self.face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
        self.ResNet50_model = ResNet50(weights='imagenet')
        self.model_builder()

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        img = preprocess_input(path_to_tensor(img_path))
        prediction = np.argmax(self.ResNet50_model.predict(img))
        return (prediction <= 268) & (prediction >= 151)

    def predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.main_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]

    def model_builder(self):
        self.main_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        self.main_model.add(Dense(133, activation='softmax'))
        self.main_model.load_weights('data/saved_models/weights.best.Resnet50.hdf5')

    def run(self, file_path):
        if self.dog_detector(file_path):
            return "DOG",self.predict_breed(file_path)
        elif self.face_detector(file_path):
            return "HUMAN",self.predict_breed(file_path)
        else:
            return "Error",None
        
if __name__ == "__main__":
    engine = Engine()
    for path in glob("data/images/*"):
        print(path, engine.prediction(path))
