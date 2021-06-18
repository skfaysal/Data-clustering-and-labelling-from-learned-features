# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

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
from cv2 import cvtColor, COLOR_BGR2RGB
from keras_efficientnets import custom_objects
from keras_efficientnets import EfficientNetB5
########################### THIS IS A Supervised clustering example  #############################

# # Image files stored in the folders
# my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
# # print(my_files_1)

my_files_1 = glob.glob('./input_img/*/*')
print(my_files_1)
print(len(my_files_1))


start = time.time()

# model = ResNet50(weights='imagenet', pooling=max, include_top=False)

# model = EfficientNetB5(weights='imagenet', pooling=max, include_top=False)

path = (str(os.getcwd())+'/models/'+'b5_newpreprocessed_full_fold4.h5')

model = load_model(path)

layer_name = 'swish_116'
# print(model.summary())

for layer in model.layers:
    print(layer.name)
####### GENERATING and Accumulating All FEATURES

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

def feature_extraction(layer_name,model,img):

    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
        
    features = intermediate_layer_model.predict(img)

    features = np.ravel(features)

    print(".......flatten features.......")
    print(features.shape)


    return features    

start = time.time()
# hyper parameters
tol = 7
IMG_SIZE = 224
sigmaX = 10

k=0
my_feature = []  # All feature is to stored here
small_file =[]
N_of_Cluster = 5

for index in range(len(my_files_1)-1): 
        

        img = load_gauss(my_files_1[index],tol,IMG_SIZE,sigmaX)

        features = feature_extraction(layer_name,model,img)
        my_feature.append(features)
        print("index ",index, "   feature_shape ",features.shape)

        # features_reduce = features.squeeze()
        # train_featues.write(' '.join(str(x) for x in features.squeeze()) + '\n')


print(my_feature)

labels =[]

def cluster (my_feature):

        print("\n\n clustering in Progress")

        kmeans = KMeans(n_clusters=N_of_Cluster)
        kmeans = kmeans.fit(my_feature)

        # Get cluster numbers for each face
        labels = kmeans.predict(my_feature)
        print(labels)

        return labels


# for (label, face) in zip(labels, faces):
#     face["group"] = int(label)

labels = cluster(my_feature)

print("\n\n labels from KMEANS:",labels)

def save_img_cluster (labels):

        target_dir = './Clustered_folder/'
        # Gather directory contents
        contents = [os.path.join(target_dir, i) for i in os.listdir(target_dir)]

        print(contents)
        # Iterate and remove each item in the appropriate manner
        [os.remove(i) if os.path.isfile(i) or os.path.islink(i) else shutil.rmtree(i) for i in contents]
        
        for i in range(N_of_Cluster):
                try :
                        os.mkdir('./Clustered_folder/'+str(i))
                except:
                        traceback.print_exc()


        for index, lb in enumerate(labels):

            img = cv2.imread(my_files_1[index])
            write_path = './Clustered_folder/' + str(lb) + '/'
            cv2.imwrite(write_path + str(index)+'.jpg',img)


save_img_cluster(labels)


end = time.time()
print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')
cv2.destroyAllWindows()
