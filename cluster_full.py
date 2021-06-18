import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import traceback
#from google.cloud import storage
from io import BytesIO
import os
import time
import glob
import cv2
from matplotlib import pyplot as plt
import scipy.misc

import shutil
from numpy import array
from PIL import Image
from keras.models import load_model
from keras.models import Model
from cv2 import cvtColor, COLOR_BGR2RGB
from keras_efficientnets import custom_objects
from keras_efficientnets import EfficientNetB5

# lr
from keras.preprocessing.image import load_img
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
########################### THIS IS A Supervised clustering example  #############################

def crop_image_from_gray(img):
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:  # image is too dark so that we crop out everything,
                return img  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    # for taking input image and do some preprocessing
def load_gauss(im_path,tol,IMG_SIZE,sigmaX):  # load_ben_color
        img = cvtColor(array(Image.open(im_path)), COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_size = (IMG_SIZE,IMG_SIZE)
        img = cv2.resize(img, target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img / 255, axis=0)
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        return img

def left_right_img(path,IMG_SIZE):
    original = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = preprocess_input(image_batch.copy())

    return processed_image

def feature_extraction(layer_name,model,img):

    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
        
    features = intermediate_layer_model.predict(img)

    features = np.ravel(features)

    print(".......flatten features.......")
    print(features.shape)


    return features    

def cluster (my_feature):

        print("\n\n clustering in Progress")

        print("\nlength of feature received",len(my_feature))

        kmeans = KMeans(n_clusters=N_of_Cluster)
        kmeans = kmeans.fit(my_feature)

        # Get cluster numbers for each face
        labels = kmeans.predict(my_feature)
        print("\nLength of lebes created out of it: ",len(labels))

        return labels


def save_img_cluster (labels,patient_id):

        target_dir = './Clustered_folder/'

       

        path = target_dir+str(patient_id)

        try:
            if not os.path.exists(path):

                os.mkdir(path)
        except:
            traceback.print_exc()

        target_dir = path+'/'

        print("\n\n\n",target_dir)

        for i in range(N_of_Cluster):
            try :
                    os.mkdir(str(target_dir)+str(i))
            except:
                    traceback.print_exc()


        for index,lb in enumerate(labels):
            print(index,lb)

            img = cv2.imread(my_files_1[index])
            write_path = str(target_dir) + str(lb) + '/'
            cv2.imwrite(write_path + str(index)+'.jpg',img)

start = time.time()


###### PATIENT CHUNK SELECTION #####
start_index = int(input("enter the starting index: "))

ending_index = int(input("enter the ending index: "))

patient_ids = sorted(os.listdir(str(os.getcwd())+'/input_img/'))[start_index-1:ending_index]

print(patient_ids)


#### LOAD MODELS AND DEFINE LAYER NAME ####
dr_path = (str(os.getcwd())+'/models/'+'b5_newpreprocessed_full_fold4.h5')

# lr_path = (str(os.getcwd())+'/models/'+'model_binary_right_left_retina_resnet50_53k.h5')

model = load_model(dr_path)

layer_name = 'swish_116'
# print(model.summary())

for layer in model.layers:
    print(layer.name)


#### HYPERPAREMETERS ####
tol = 7
IMG_SIZE = 224
sigmaX = 10
k=0
N_of_Cluster = 4



for patient_id in patient_ids:

    print("\n......[INFO]Clustering for "+str(patient_id)+" is started......")
    my_files_1 = glob.glob(str(os.getcwd())+'/input_img/'+str(patient_id)+'/'+'*')

    print(len(my_files_1))

    my_feature = []
    for index in range(len(my_files_1)):

        img = load_gauss(my_files_1[index],tol,IMG_SIZE,sigmaX)

        features = feature_extraction(layer_name,model,img)

        my_feature.append(features)
        print("index ",index, "   feature_shape ",features.shape)


   
    #### CREATE CLUSTERS AND RETURNS LIST OF LABELS ####

    labels = cluster(my_feature)

    print("\n\n labels from KMEANS:",labels)

    #### SAVE CLUSTERD IMAGES ####

    save_img_cluster(labels,patient_id)

    end = time.time()
    print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')
